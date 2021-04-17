import selectors
import struct
import threading

from Common.src.network.message import Message, OwnedMessage
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
        self.is_read_header_error = False
        self.is_read_body_error = False

    def read_message(self, sock):
        if self.is_server:
            msg = OwnedMessage(sock)
        else:
            msg = Message()
        msg.header = self.read_header(sock)
        if not self.is_read_header_error:
            msg.body = self.read_body(sock, msg.get_body_size())
            if not self.is_read_body_error:
                self.messages_in.append(msg)
        self.is_read_header_error = False
        self.is_read_body_error = False

    def read_header(self, sock):
        header_len = 8
        msg_header = b''
        try:
            msg_header = sock.recv(header_len)
            if msg_header:
                return msg_header
            else:
                self.is_read_header_error = True
                self.error_callback(sock, "Empty message header.")
        except Exception:
            self.is_read_header_error = True
            self.error_callback(sock, "Message header read failed.")

    def read_body(self, sock, size):
        msg_body = b''
        try:
            while len(msg_body) < size:
                bytes_left = size - len(msg_body)
                if bytes_left < 4096:
                    msg_body += sock.recv(bytes_left)
                else:
                    msg_body += sock.recv(4096)
            # print("Bytesize of message body received: ", len(msg_body))
            if msg_body:
                return msg_body
            else:
                self.is_read_body_error = True
                self.error_callback(sock, "Empty message body.")
        except Exception:
            self.is_read_body_error = True
            self.error_callback(sock, "Message body read failed.")

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

    def send_message(self, sock, msg):
        msg = OwnedMessage(sock, msg.header, msg.body)
        self.messages_out.append(msg)
        if not self.writing_msg:
            self.wm_mutex.acquire()
            self.writing_msg = True
            self.wm_mutex.release()
            write_thread = threading.Thread(target=self.write_message)
            write_thread.start()
            # self.write_message()
