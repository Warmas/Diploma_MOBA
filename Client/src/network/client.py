import socket
import selectors
import threading
import traceback

from Common.src.network.tsdeque import TsDeque
from Common.src.network.connection import Connection
from Common.src.network.message import Message


class Client:
    def __init__(self, process_message_callback):
        self.server_port = 54321
        self.server_address = "127.0.0.1"
        self.sel = selectors.DefaultSelector()
        self.sock = None
        self.connection_object = None
        self.messages_in = TsDeque()
        self.exc_started = False
        self.exc_mutex = threading.Lock()

        self.process_message_callback = process_message_callback

    def start_connection(self, address, port):
        self.server_port = port
        self.server_address = address
        server_addr = (self.server_address, self.server_port)
        print("Connecting to server: ", server_addr)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)
        self.sock.connect_ex(server_addr)
        self.sel.register(self.sock, selectors.EVENT_READ | selectors.EVENT_WRITE, data=None)
        self.connection_object = Connection(self.messages_in, self.sel, False, self.connection_error)
        event_thread = threading.Thread(target=self.start_event_thread)
        event_thread.start()

    def start_event_thread(self):
        self.connection_object.is_connected = True
        while self.connection_object.is_connected:
            events = self.sel.select(timeout=None)
            for key, mask in events:
                sock = key.fileobj
                if mask & selectors.EVENT_READ:
                    self.connection_object.read_message(sock)

    def connection_error(self, sock, msg):
        self.exc_mutex.acquire()
        if not self.exc_started:
            self.exc_started = True
            self.exc_mutex.release()
            self.connection_object.is_connected = False
            self.sel.unregister(sock)
            sock.close()
            print("Closing connection due to error!", "\nInfo: ", msg)
            # print(traceback.format_exc())
        else:
            self.exc_mutex.release()

    def process_all_messages(self):
        while not self.messages_in.empty():
            msg = self.messages_in.front()
            self.messages_in.pop_left()
            self.process_message_callback(msg)

    def send_message(self, msg_id, msg_body, is_body_string=False):
        if self.connection_object.is_connected:
            msg = Message()
            msg.set_message(msg_id, msg_body, is_body_string)
            self.connection_object.send_message(self.sock, msg)
        else:
            print("Connection is closed, message send failed!")

    def get_connection_state(self):
        return self.connection_object.is_connected
