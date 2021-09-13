from enum import Enum
import struct


class MessageTypes(Enum):
    PingServer = 1
    MessagePrint = 2
    Authentication = 3
    NewPlayer = 4
    CastSpell = 5
    PlayerMoveTo = 6
    RemoveGameObject = 7
    UpdateHealth = 8
    MobsMoveTo = 9
    MobsKilled = 10
    HealPlaceChange = 11
    PauseGame = 12
    ClientReady = 13
    StartGame = 14
    ResetMap = 15
    ContinueGame = 16
    TransitionData = 17
    TransferDone = 18
    OptimizeDone = 19
    CloseGame = 20


class Message:
    def __init__(self, header=b'00000000', body=b''):
        self.header = header
        self.body = body

    def set_header_by_id(self, msg_id):
        msg_id_b = struct.pack("!i", msg_id)
        self.header = msg_id_b + self.header[4:]

    def set_body(self, msg_body, is_string=False):
        if is_string:
            msg_body = msg_body.encode("utf-8")
        self.body = msg_body
        body_size_b = struct.pack("!i", len(self.body))
        self.header = self.header[:4] + body_size_b

    def set_message(self, msg_id, msg_body, is_string_body=False):
        self.set_header_by_id(msg_id)
        self.set_body(msg_body, is_string_body)

    def get_msg_id(self):
        msg_id = struct.unpack("!i", self.header[:4])[0]
        return msg_id

    def get_body_size(self):
        body_size = struct.unpack("!i", self.header[4:])[0]
        return body_size

    def get_body(self):
        return self.body

    def get_body_as_string(self):
        msg_body = self.body.decode("utf-8")
        return msg_body

    def push_int(self, num):
        int_data = struct.pack("!i", num)
        self.body += int_data
        body_size_b = struct.pack("!i", len(self.body))
        self.header = self.header[:4] + body_size_b

    def get_int(self):
        # We don't need to update body size
        num = struct.unpack("!i", self.body[:4])[0]
        self.body = self.body[4:]
        return num

    def push_float(self, num):
        float_data = struct.pack("!f", num)
        self.body += float_data
        body_size_b = struct.pack("!i", len(self.body))
        self.header = self.header[:4] + body_size_b

    def get_float(self):
        num = struct.unpack("!f", self.body[:4])[0]
        self.body = self.body[4:]
        return num

    def push_string(self, string):
        length = struct.pack("!i", len(string))
        self.body += length
        self.body += string.encode("utf-8")
        body_size_b = struct.pack("!i", len(self.body))
        self.header = self.header[:4] + body_size_b

    def get_string(self):
        string_length = self.get_int()
        string_data = self.body[:string_length]
        self.body = self.body[string_length:]
        string = string_data.decode("utf-8")
        return string

    def push_bytes(self, bytes_to_push):
        self.body += bytes_to_push
        body_size_b = struct.pack("!i", len(self.body))
        self.header = self.header[:4] + body_size_b

    def push_double(self, num):
        double_data = struct.pack("!d", num)
        self.body += double_data
        body_size_b = struct.pack("!i", len(self.body))
        self.header = self.header[:4] + body_size_b

    def get_double(self):
        num = struct.unpack("!d", self.body[:8])[0]
        self.body = self.body[8:]
        return num


class OwnedMessage(Message):
    def __init__(self, socket, header=b'00000000', body=b''):
        super(OwnedMessage, self).__init__(header, body)
        self.socket = socket


def push_int(my_bytes, num):
    int_data = struct.pack("!i", num)
    my_bytes += int_data
    return my_bytes


def push_float(my_bytes, num):
    float_data = struct.pack("!f", num)
    my_bytes += float_data
    return my_bytes


def push_string(my_bytes, string):
    length = struct.pack("!i", len(string))
    my_bytes += length
    my_bytes += string.encode("utf-8")
    return my_bytes

