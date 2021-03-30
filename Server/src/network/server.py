import socket
import selectors
import threading
import traceback

from Common.src.network.tsdeque import TsDeque
from Common.src.network.connection import Connection


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
        self.connection_object = Connection(self.messages_in, self.sel, True, self.remove_client)
        event_thread = threading.Thread(target=self.start_event_thread)
        event_thread.start()
        print("Server started at: " + str(self.port) + " : " + self.address)

    def start_event_thread(self):
        self.connection_object.is_connected = True
        should_stop = False
        while not should_stop:
            events = self.sel.select(timeout=None)
            for key, mask in events:
                sock = key.fileobj
                # New connection
                if key.data is None:
                    self.accept_connection(sock)
                # Message from connected
                else:
                    try:
                        if mask & selectors.EVENT_READ:
                            self.connection_object.read_message(sock)
                    except Exception:
                        should_stop = self.remove_client(sock, "Connection lost.")

    def accept_connection(self, sock):
        conn, addr = sock.accept()
        self.connection_list.append(conn)
        print("Connection accepted for: ", addr)
        conn.setblocking(False)
        self.sel.register(conn, selectors.EVENT_READ | selectors.EVENT_WRITE, data="msg")

    def remove_client(self, sock, msg):
        print("Closing connection due to error, for address: ", sock.getpeername(), "\nInfo: ", msg)
        self.connection_lost_callback(sock)
        self.connection_list.remove(sock)
        self.sel.unregister(sock)
        sock.close()
        #print(traceback.format_exc())
        if not len(self.sel.get_map()):
            return True
        else:
            return False

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

    def get_connections_n(self):
        return len(self.connection_list)
