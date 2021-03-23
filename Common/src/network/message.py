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
    GameState = 13
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
    def __init__(self, socket, msg_header, msg_body):
        self.socket = socket
        self.header = msg_header
        self.body = msg_body
