import sys

from AI_Client.src.main_ai_client import start_ai_client

# Environment requirements: numpy, pytorch, gym, openGL, (freeGlut), pyGlm

if __name__ == '__main__':
    client_id = "AI_Ben_pycharm"
    is_training = True
    is_displayed = True
    if len(sys.argv) > 2:
        client_id = sys.argv[1]
        if sys.argv[2] == "0":
            is_training = False
        if sys.argv[3] == "0":
            is_displayed = False
    start_ai_client(client_id, is_training, is_displayed)
