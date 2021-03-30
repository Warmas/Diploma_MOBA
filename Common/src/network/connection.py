import selectors
import struct
import threading

from Common.src.network.message import Message, MessageTypes, OwnedMessage, BytesMessage, MessageToSend
from Common.src.network.tsdeque import TsDeque


class Connection:
    def __init__(self, messages_in, selector, is_server, error_callback):
        self.messages_in = messages_in
        self.messages_out = TsDeque()
        self.sel = selector
        self.is_server = is_server
        self.writing_msg = False
        self.wm_mutex = threading.Lock()
        self.is_connected = False  # Probably should be mutexed too
        self.error_callback = error_callback

    def read_message(self, sock):
        msg_id, body_size = self.read_header(sock)
        is_bytes = bool(msg_id == MessageTypes.Image.value)
        msg_body = self.read_body(sock, body_size, not is_bytes)
        if is_bytes:
            msg = BytesMessage(sock, msg_id, msg_body)
        elif self.is_server:
            msg = OwnedMessage(sock, msg_id, msg_body)
        else:
            msg = Message(msg_id, msg_body)
        self.messages_in.append(msg)

    def read_header(self, sock):
        header_len = 8
        msg_header = b''
        try:
            msg_header = sock.recv(header_len)
        except Exception:
            self.error_callback(sock, "Message header read failed.")
        if msg_header:
            msg_id = struct.unpack("!i", msg_header[:4])[0]
            body_size = struct.unpack("!i", msg_header[4:])[0]
            return msg_id, body_size
        else:
            self.error_callback(sock, "Empty message header.")

    def read_body(self, sock, size, is_string):
        msg_body = b''
        try:
            while len(msg_body) < size:
                bytes_left = size - len(msg_body)
                if bytes_left < 4096:
                    msg_body += sock.recv(bytes_left)
                else:
                    msg_body += sock.recv(4096)
            # print("Bytesize of message body received: ", len(msg_body))
        except Exception:
            self.error_callback(sock, "Message body read failed.")
        if msg_body:
            if is_string:
                msg_body = msg_body.decode("utf-8")
            return msg_body
        else:
            self.error_callback(sock, "Empty message body.")

    def write_message(self):
        msg = self.messages_out.front()
        data = msg.header + msg.body
        # print('Sending to:', msg.socket.getpeername())
        while len(data) and self.is_connected:
            events = self.sel.select(timeout=None)
            for key, mask in events:
                sock = key.fileobj
                try:
                    if mask & selectors.EVENT_WRITE:
                        if sock == msg.socket:
                            sent = msg.socket.send(data)
                            data = data[sent:]
                            # print("Number of bytes sent: ", sent)
                except Exception:
                    self.error_callback(sock, "Message write failed.")
                    data = b''
        self.messages_out.pop_left()
        if not self.messages_out.empty():
            self.write_message()
        else:
            self.wm_mutex.acquire()
            self.writing_msg = False
            self.wm_mutex.release()

    def send_message(self, sock, msg_id, msg_body):
        msg_header, msg_body = OwnedMessage(sock, msg_id, msg_body).encode()
        msg = MessageToSend(sock, msg_header, msg_body)
        self.messages_out.append(msg)
        if not self.writing_msg:
            self.wm_mutex.acquire()
            self.writing_msg = True
            self.wm_mutex.release()
            write_thread = threading.Thread(target=self.write_message)
            write_thread.start()
            # self.write_message()

    def send_bytes(self, sock, msg_id, msg_body):
        writing_msg = not self.messages_out.empty()
        msg_header, msg_body = BytesMessage(sock, msg_id, msg_body).encode()
        msg = MessageToSend(sock, msg_header, msg_body)
        self.messages_out.append(msg)
        if not writing_msg:
            self.write_message()
