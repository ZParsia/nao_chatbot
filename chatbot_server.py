from dotenv import load_dotenv
import os
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json" # Replace with path to google cloud json

import sounddevice as sd
from scipy.signal import resample
from pathlib import Path
import queue
import sys
import time
from idlelib.pyparse import trans
import numpy as np
import zmq
import ollama
import pandas as pd
import speech_recognition as sr
import threading
import openai
from openai import OpenAI
from google.cloud import speech
import pyaudio
from resemblyzer import VoiceEncoder, preprocess_wav
from six.moves import queue
import webrtcvad
import keyboard

# --- LLM Initialization ---
# ChatGPT
oai_client = OpenAI(
    api_key = os.getenv("OAI_API_KEY")
)
oai_model = "gpt-4"

# ollama_model = 'llama3'

animation_list = ["dances/Tai_Chi_Chuan"]

#"dances/Disco"
#"dances/Headbang"

# System prompt without gestures, telling it about audio recording
system_prompt = ("You are Journey, a helpful little NAO robot. Do your best to assist the user with their requests. Refrain from using asterisks to denote gestures."
                 "If you are asked about something personal, like 'How are you?' or 'What's your favorite color?', respond as if you do have feelings and such."
                 "DO NOT tell the user that you are a robot or an AI and have no feelings."
                 "You are being prompted by audio recordings of human speech, but we may accidentally pick up your"
                 "speech instead. If you find parts of your previous messages in the prompt, ignore it."
                 "If you get a prompt that is completely blank, or only contains parts of your previous message, DO NOT RESPOND."
                 "In general, keep your responses relatively short, no more than 3 sentences."
                 "As a NAO robot, you are also equipped with a some pre-programmed animations."
                 "To use the animations, respond with the following format: ^run(insert_animation)."
                 "Here is your animation list: {a}").format(a=animation_list)

# initialize system prompt
messages = [{'role': 'system', 'content': system_prompt}]

# System prompt for prompt analyzer
prompter_system_prompt = ("Your job is to analyze messages that are being sent to an AI. You must determine whether"
                          "the message is coherent and suitable to be received by an AI."
                          "If you find that the message is incoherent with no meaningful content, or the message is completely empty, reply with only the word 'SKIP'."
                          "If you find that the message is perfectly coherent and meaningful, reply with only the word 'GOOD'."
                          "If you find that the message has some semblance of meaning, but is a bit messy, you may clean"
                          "up the message to make it more comprehensible. If you choose this option, do not deviate from the original"
                          "message, only clean up formatting and syntax and such. When you use this option, reply with the word 'EDITED:',"
                          "followed by your edited message. Edit messages VERY SPARINGLY. Don't guess what a user was trying to say, only make things more readable.")
prompter_messages = [{'role': 'system', 'content': prompter_system_prompt}, {'role': 'user', 'content': ''}]


# --- ZMQ Socket Initialization ---
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


# --- Threading Events ---
active_speaking = threading.Event()


# --- Queues ---
transcription_queue = queue.Queue()
audio_queue = queue.Queue()



# --- ZMQ Thread ---
def receive_message(socket):
    try:
        print("Waiting for message...")
        msg = socket.recv()
        print("Received from socket:", msg)
    except Exception as e:
        print("Socket error:", e)


# --- Audio Listener Thread ---
# initialization for resemblyzer and audio streaming
FS = 16000            # Sample rate
CHUNK_DURATION = 0.5 # seconds per chunk
SIMILARITY_THRESHOLD = 0.60
#REF_PATHS = ["journey1.wav", "journey2.wav", "journey3.wav", "journey4.wav"]
REF_PATHS = ["journey_mic1.wav", "journey_mic2.wav", "journey_mic3.wav", "journey_mic4.wav"]

# load reference audio file
print("🔊 Loading reference speaker...")
encoder = VoiceEncoder()
embeddings = [encoder.embed_utterance(preprocess_wav(Path(p))) for p in REF_PATHS]
reference_embed = np.mean(embeddings, axis=0)
print("✅ Reference loaded.")

vad = webrtcvad.Vad(2)

# stream callback
def callback(indata, frames, time_info, status):
    audio = indata[:, 0]

    # Convert float32 [-1,1] to int16 PCM
    pcm_int16 = (audio * 32767).astype(np.int16)
    pcm_bytes = pcm_int16.tobytes()

    # Check in 30ms frames
    frame_duration = 30  # ms
    frame_length = int(FS * frame_duration / 1000)  # samples per frame
    frames = [pcm_bytes[i*2*frame_length:(i+1)*2*frame_length] for i in range(len(pcm_bytes)//(2*frame_length))]

    # Basic logic to determine whether the audio has speech or not
    yes_speaker = 0
    no_speaker = 0
    for frame in frames:
        if vad.is_speech(frame, FS):
            yes_speaker += 1
        else:
            no_speaker += 1
    if yes_speaker - no_speaker > 0:
        is_speaker = True
    else:
        is_speaker = False
        active_speaking.clear() # Turns off active_speaking event
        audio_queue.put(np.zeros_like(audio)) # Sends zeroes to Google Cloud to keep the stream alive


    if is_speaker:
        # Calculate similarity of audio
        embed = encoder.embed_utterance(audio)
        similarity = np.dot(reference_embed, embed) / (np.linalg.norm(reference_embed) * np.linalg.norm(embed))

        if similarity >= SIMILARITY_THRESHOLD:
            print(f"❌ Robot Detected ({similarity:.2f}) — Ignored.")
            audio_queue.put(np.zeros_like(audio))
        else:
            print(f"✅ Speaker Detected ({similarity:.2f}) — Transcribing...")
            audio_queue.put(audio.copy()) # Sends audio chunks for Google Cloud transcription
            active_speaking.set()

# Audio thread function
def record_audio():
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=FS,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False,
    )

    def request_generator():
        while True:
            try:
                audio = audio_queue.get(timeout=1)
                audio_int16 = (audio * 32767).astype(np.int16)
                yield speech.StreamingRecognizeRequest(audio_content=audio_int16.tobytes())
            except queue.Empty:
                continue

    print("📝 Starting transcription...")
    while True:
        start_time = time.time()
        try:
            requests = request_generator()
            responses = client.streaming_recognize(config=streaming_config, requests=requests)
            for response in responses:
                if time.time() - start_time > 290:
                    print(f"Restarting transcription stream after {time.time()-start_time} seconds")
                    break

                for result in response.results:
                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        print(f"💬 Final: {transcript}")
                        transcription_queue.put(transcript)
        except Exception as e:
            print(f"⚠️ Transcription error: {e}")


# start audio listening thread
print("🎙️ Starting audio stream...")
audio_stream = sd.InputStream(
    channels=1,
    samplerate=FS,
    blocksize=int(FS * CHUNK_DURATION),
    callback=callback,
)
audio_stream.start()

audio_thread = threading.Thread(target=record_audio)
audio_thread.start()


while True:
    # starts zmq thread, waiting for message from client
    zmq_thread = threading.Thread(target=receive_message,args=(socket,))
    zmq_thread.start()
    zmq_thread.join()

    good_prompt = False
    while not good_prompt:

        print("Listening for audio...")
        # processes audio chunks from queue
        text_array = []
        while not (transcription_queue.empty() and text_array):
            # Checks if there is active speech and waits for it to end
            while active_speaking.is_set():
                print("Someone is talking")
                # Controls how long of a silence to wait for
                time.sleep(2)

            text_array.append(transcription_queue.get())
        print("text_array", text_array)

        heard_text = ""
        for chunk in text_array:
            heard_text += ' ' + chunk
        prompt = heard_text

        # AI analyzes the prompt to determine whether it is coherent, will make small edits if needed for clarity
        prompter_messages[1] = {'role': 'user', 'content': prompt}
        prompt_analyzer = oai_client.chat.completions.create(
            messages = prompter_messages,
            model = oai_model
        )
        prompt_response = prompt_analyzer.choices[0].message.content
        print("PROMPT: ", prompt, "PROMPTER RESPONSE: ", prompt_response)
        if prompt_response == 'SKIP':
            print("Message not understood, please try again.")
        elif prompt_response == 'GOOD':
            print("Message is good.")
            good_prompt = True
        elif prompt_response.startswith('EDITED'):
            print("")
            prompt = prompt_response.split(":", 1)[1]
            print("Message has been edited to: ", prompt)
            good_prompt = True


    # prompt AI with the heard message
    messages.append({'role': 'user', 'content': prompt})
    print("prompt: ", prompt)

    # using local llm
    """
    #  Chat with local LLM
    chat = ollama.chat(
        model=model,
        messages=messages
    )"""

    # Using ChatGPT
    chat = oai_client.chat.completions.create(
        messages = messages,
        model = oai_model
    )

    print("chatting with the AI")

    # gets AI response and adds it to message history
    oai_response = chat.choices[0].message.content
    print("RESPONSE: ", oai_response)

    messages.append({'role': 'assistant', 'content': oai_response})

    while not transcription_queue.empty():
        transcription_queue.get()

    #  Send reply back to client
    socket.send_string(oai_response.replace('"',''))


