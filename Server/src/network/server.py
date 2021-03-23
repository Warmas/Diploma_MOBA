import socket
import selectors
import threading

from Common.src.network.tsdeque import TsDeque
from Common.src.network.connection import Connection

# https://realpython.com/python-sockets/
# Events


class Server:
    def __init__(self, process_message_callback, connection_lost_callback):
        self.address = "127.0.0.1"
        self.port = 54321
        self.sel = selectors.DefaultSelector()
        self.connection_object = None  # General connection with the outside world
        self.connection_list = []
        self.messages_out = TsDeque()
        self.messages_in = TsDeque()
        self.updates_out = []

        self.process_message_callback = process_message_callback
        self.connection_lost_callback = connection_lost_callback

    def start(self):
        conn_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.address = socket.gethostname()
        conn_sock.bind((self.address, self.port))
        conn_sock.listen()
        conn_sock.setblocking(False)
        self.sel.register(conn_sock, selectors.EVENT_READ, data=None)
        self.connection_object = Connection(self.messages_in, self.sel, True)
        event_thread = threading.Thread(target=self.start_event_thread)
        event_thread.start()
        print("Server started at: " + str(self.port) + " : " + self.address)

    def start_event_thread(self):
        while True:
            events = self.sel.select(timeout=None)
            for key, mask in events:
                # New connection
                if key.data is None:
                    self.accept_connection(key.fileobj)
                # Message from connected
                else:
                    try:
                        self.connection_object.read_message(key, mask)
                    except Exception:
                        self.remove_client(key.fileobj)

    def accept_connection(self, sock):
        conn, addr = sock.accept()
        self.connection_list.append(conn)
        print("Connection accepted for: ", addr)
        conn.setblocking(False)
        self.sel.register(conn, selectors.EVENT_READ | selectors.EVENT_WRITE, data="msg")

    def remove_client(self, sock):
        print("Closing connection due to error, for address: ", sock.getpeername(), "\nInfo: connection lost")
        self.connection_lost_callback(sock)
        self.connection_list.remove(sock)
        self.sel.unregister(sock)
        sock.close()
        # print('Error: exception for', f'{message.addr}:\n{traceback.format_exc()}')

    def process_all_messages(self):
        if not self.messages_in.empty():
            msg = self.messages_in.front()
            self.process_message_callback(msg)
            self.messages_in.pop_left()
            self.process_all_messages()

    def message_all(self, msg_id, msg_body):
        for sock in self.connection_list:
            self.connection_object.send_message(sock, msg_id, msg_body)

    def message_all_but_one(self, ignore, msg_id, msg_body):
        for sock in self.connection_list:
            if not sock == ignore:
                self.connection_object.send_message(sock, msg_id, msg_body)

    def send_updates(self):
        for message in self.updates_out:
            self.message_all(message)
        self.updates_out.clear()

    def send_message(self, sock, message_id, message_data):
        self.connection_object.send_message(sock, message_id, message_data)

    def send_bytes(self, sock, message_id, message_data):
        self.connection_object.send_bytes(sock, message_id, message_data)
