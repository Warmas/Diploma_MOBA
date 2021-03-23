import socket
import selectors
import threading

from Common.src.network.tsdeque import TsDeque
from Common.src.network.connection import Connection


class Client:
    def __init__(self, process_message_callback):
        self.server_port = 54321
        self.server_address = "127.0.0.1"
        self.sel = selectors.DefaultSelector()
        self.sock = None
        self.connection_object = None
        self.messages_out = TsDeque()
        self.messages_in = TsDeque()
        self.connection_ready = False

        self.process_message_callback = process_message_callback

    def start_connection(self, address, port):
        self.server_port = port
        self.server_address = address
        server_addr = (self.server_address, self.server_port)
        print("Connecting to server: ", server_addr)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setblocking(False)
        self.sock.connect_ex(server_addr)
        self.sel.register(self.sock, selectors.EVENT_READ, data=None)
        self.connection_object = Connection(self.messages_in, self.sel, False)
        event_thread = threading.Thread(target=self.start_event_thread)
        event_thread.start()
        self.connection_ready = True

    def start_event_thread(self):
        while True:
            events = self.sel.select(timeout=None)
            for key, mask in events:
                try:
                    self.connection_object.read_message(key, mask)
                except Exception:
                    print("Closing connection due to error, for address: ", key.fileobj, "\nInfo: connection lost")
                    self.sel.unregister(key.fileobj)
                    key.fileobj.close()
                    # print('Error: exception for', f'{message.addr}:\n{traceback.format_exc()}')

    def process_all_messages(self):
        if not self.messages_in.empty():
            msg = self.messages_in.front()
            self.process_message_callback(msg)
            self.messages_in.pop_left()
            self.process_all_messages()

    def send_message(self, message_id, message_data):
        self.connection_object.send_message(self.sock, message_id, message_data)

    def send_bytes(self, message_id, message_data):
        self.connection_object.send_bytes(self.sock, message_id, message_data)
