# -*- coding: utf-8 -*-

import zmq
from naoqi import ALProxy

DESTINY = "192.168.1.69"
SAM = "192.168.1.128"
JOURNEY = "192.168.1.55"
ANGEL = "192.168.1.35"
ROBOT_IP = JOURNEY
PORT = 9559

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

tts = ALProxy("ALAnimatedSpeech", ROBOT_IP, PORT)
#welcome_message = "Hello everyone, and welcome to the Collaborative Robotics Lab. My name is Journey, how can I assist you today?"
#tts.say(welcome_message)


cancel = False

while True:

    send_request = "Continue"

    print("Sending request ...")
    socket.send(send_request)

    try:
        print("Waiting for message... Press Ctrl+C to cancel.")
        while True:
            events = dict(poller.poll(100))  # Poll with short timeout (100ms)
            if socket in events:
                message = socket.recv()

                print("Received reply {m}".format(m=message))

                # print("MESSAGE: {m}".format(m=message))

                tts.say(message)

                break

    except KeyboardInterrupt:
        print("Cancelled by user.")
        cancel = True

    if cancel:
        break
