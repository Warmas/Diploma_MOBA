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
    ResetMap = 13
    ContinueGame = 14
    Image = 15
    TransitionData = 16
    TransferDone = 17
    ClientReady = 18
    StartGame = 19


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
        something = "asd"

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


class OwnedMessage(Message):
    def __init__(self, socket, header=b'00000000', body=b''):
        super(OwnedMessage, self).__init__(header, body)
        self.socket = socket
