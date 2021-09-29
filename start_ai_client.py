import sys
import argparse

from AI_Client.src.main_ai_client import start_ai_client

# Environment requirements: numpy, pytorch, openGL, (freeGlut), pyGlm

if __name__ == '__main__':
    client_id = "AI_Ben_pycharm"
    is_training = False
    is_displayed = True
    is_load_weights = False
    weight_file = "last_agent_weight.pth"
    optimizer_file = "last_optimizer.pth"

    if len(sys.argv) > 1:
        help_msg = "Help message"
        parser = argparse.ArgumentParser(description=help_msg)
        parser.add_argument("-i", "--client_id", type=str, required=True, help="Unique name of the client.")
        parser.add_argument("-t", "--is_training", type=bool, help="Is training optimization to be run.")
        parser.add_argument("-d", "--no_display", type=bool, help="Turns of display.")
        parser.add_argument("-w", "--weights", type=str, help="File name of the weights file to be loaded.")
        parser.add_argument("-o", "--optimizer", type=str, help="File name of the optimizer file to be loaded.")
        args = parser.parse_args()

        client_id = args.client_id
        if args.is_training:
            is_training = True
        if args.no_display:
            is_displayed = False
        if args.weights:
            is_load_weights = True
            weight_file = args.weights
        if args.optimizer:
            optimizer_file = args.optimizer

    start_ai_client(client_id, is_training, is_displayed, is_load_weights, weight_file, optimizer_file)
