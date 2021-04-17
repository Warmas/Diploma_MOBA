import sys

from Client.src.main_client import start_client


if __name__ == '__main__':
    client_id = "Ben_pycharm"
    if len(sys.argv) > 1:
        client_id = sys.argv[1]
    start_client(client_id)
