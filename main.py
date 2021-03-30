import Client.src.main_client
import numpy as np
import math
import torch
import struct
import sys

if __name__ == '__main__':
    my_bytes1 = struct.pack("!i", 0)
    my_bytes2 = b'1111'
    my_bytes = my_bytes1 + my_bytes2
    result = struct.unpack("!i", my_bytes[0])
    print(result)