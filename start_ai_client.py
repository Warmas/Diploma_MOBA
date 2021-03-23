import sys

from AI_Client.src.main_ai_client import start_ai_client


if __name__ == '__main__':
    client_id = "AI_Ben_pycharm"
    is_training = True
    if len(sys.argv) > 2:
        client_id = sys.argv[1]
        if sys.argv[2] == "1":
            is_training = True
        else:
            is_training = False
    start_ai_client(client_id, is_training)
