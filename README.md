# nao_chatbot
Facilitates human speech communication with NAO robots

# Usage
Requires OpenAI and Google Cloud API keys

Use these directions to get your Google Cloud key:

https://stackoverflow.com/questions/46287267/how-can-i-get-the-file-service-account-json-for-google-translate-api

Once these keys are obtained, they will need to be inserted into chatbot_server.py.

To operate the system, run chatbot_server.py in your IDE with the latest python version installed, then run chatbot_client.py in your nao anaconda environment with python 2.7. Audio will be recorded through your devices microphone system and given to the LLM to generate NAO's response. This behavior will loop until interrupted by the user.
