import selectors
import struct

from Common.src.network.message import *
from Common.src.network.tsdeque import TsDeque


class Connection:
    def __init__(self, messages_in, selector, is_server):
        self.messages_in = messages_in
        self.messages_out = TsDeque()
        self.sel = selector
        self.is_server = is_server

    def read_message(self, key, mask):
        sock = key.fileobj
        if mask & selectors.EVENT_READ:
            msg_id, body_size = self.read_header(sock)
            is_image = bool(msg_id == MessageTypes.Image.value)
            msg_body = self.read_body(sock, body_size[0], not is_image)
            if is_image:
                msg = BytesMessage(sock, msg_id[0], msg_body)
            elif self.is_server:
                msg = OwnedMessage(sock, msg_id[0], msg_body)
            else:
                msg = Message(msg_id[0], msg_body)
            self.messages_in.append(msg)

    def read_header(self, sock):
        header_len = 8
        msg_header = sock.recv(header_len)
        if msg_header:
            msg_id = struct.unpack("!i", msg_header[:4])
            body_size = struct.unpack("!i", msg_header[4:])
            return msg_id, body_size
        else:
            print("Closing connection due to error, for address: ", sock.getpeername(), "\nInfo: Empty header", )
            self.sel.unregister(sock)
            sock.close()

    def read_body(self, sock, size, is_string):
        msg_body = sock.recv(size)
        if msg_body:
            if is_string:
                msg_body = msg_body.decode("utf-8")
            return msg_body
        else:
            print("Closing connection due to error, for address: ", sock.getpeername(), "\nInfo: Empty body", )
            self.sel.unregister(sock)
            sock.close()

    def write_message(self):
        msg = self.messages_out.front()
        # print('Sending to:', msg.socket.getpeername())
        msg.socket.send(msg.header + msg.body)
        self.messages_out.pop_left()
        if not self.messages_out.empty():
            self.write_message()

    def send_message(self, sock, msg_id, msg_body):
        writing_msg = not self.messages_out.empty()
        msg_header, msg_body = OwnedMessage(sock, msg_id, msg_body).encode()
        msg = MessageToSend(sock, msg_header, msg_body)
        self.messages_out.append(msg)
        if not writing_msg:
            self.write_message()

    def send_bytes(self, sock, msg_id, msg_body):
        writing_msg = not self.messages_out.empty()
        msg_header, msg_body = BytesMessage(sock, msg_id, msg_body).encode()
        msg = MessageToSend(sock, msg_header, msg_body)
        self.messages_out.append(msg)
        if not writing_msg:
            self.write_message()
