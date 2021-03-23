import Client.src.main_client
import numpy as np
import math
import torch
import struct
import sys

if __name__ == '__main__':
    client_id = "AI_Ben_pycharm"
    is_training = True
    if len(sys.argv) > 2:
        client_id = sys.argv[1]
        if sys.argv[2] == "1":
            is_training = True
        else:
            is_training = False
    print(is_training)