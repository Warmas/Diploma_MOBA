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
    def __init__(self, msg_id, msg_body):
        self.id = msg_id
        self.body = msg_body

    def encode(self):
        body_bytes = self.body.encode("utf-8")
        body_size = struct.pack("!i", len(body_bytes))
        msg_id = struct.pack("!i", self.id)
        header = msg_id + body_size
        return header, body_bytes


class OwnedMessage:
    def __init__(self, socket, msg_id, msg_body):
        self.socket = socket
        self.id = msg_id
        self.body = msg_body

    def encode(self):
        body_bytes = self.body.encode("utf-8")
        body_size = struct.pack("!i", len(body_bytes))
        msg_id = struct.pack("!i", self.id)
        header = msg_id + body_size
        return header, body_bytes


class BytesMessage(OwnedMessage):
    def __init__(self, socket, msg_id, msg_body):
        super(BytesMessage, self).__init__(socket, msg_id, msg_body)

    def encode(self):
        body_size = struct.pack("!i", len(self.body))
        msg_id = struct.pack("!i", self.id)
        header = msg_id + body_size
        return header, self.body


class MessageToSend:
    """Contains the bytes of the message."""
    def __init__(self, socket, msg_header, msg_body):
        self.socket = socket
        self.header = msg_header
        self.body = msg_body


class NewMessage:
    def __init__(self, header=b'', body=b''):
        self.header = header
        self.body = body

    def set_header(self, msg_id):
        msg_id_b = struct.pack("!i", msg_id)
        body_size_b = struct.pack("!i", len(self.body))
        self.header = msg_id_b + body_size_b

    def set_body(self, msg_body, is_string=False):
        if is_string:
            msg_body = msg_body.encode("utf-8")
        self.body = msg_body

    def set_message(self, msg_id, msg_body, is_string=False):
        self.set_header(msg_id)
        self.set_body(msg_body, is_string)

    def get_msg_id(self):
        msg_id = struct.unpack("!i", self.header[:4])[0]
        return msg_id

    def get_msg_size(self):
        body_size = struct.unpack("!i", self.header[4:])[0]
        return body_size
        

class NewOwnedMessage(NewMessage):
    def __init__(self, socket, header=b'', body=b''):
        super(NewOwnedMessage, self).__init__(header, body)
        self.socket = socket
