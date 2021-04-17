import socket
import selectors
import threading
import traceback

from Common.src.network.tsdeque import TsDeque
from Common.src.network.connection import Connection
from Common.src.network.message import Message


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
        self.should_stop = False

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

    def stop(self):
        self.should_stop = True

    def start_event_thread(self):
        self.connection_object.is_connected = True
        while not self.should_stop:
            events = self.sel.select(timeout=1)
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
                        self.remove_client(sock, "Connection lost.")

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

    def process_all_messages(self):
        while not self.messages_in.empty():
            msg = self.messages_in.front()
            self.messages_in.pop_left()
            self.process_message_callback(msg)

    def send_message(self, sock, msg_id, msg_body, is_body_string=False):
        msg = Message()
        msg.set_message(msg_id, msg_body, is_body_string)
        self.connection_object.send_message(sock, msg)

    def message_all(self, msg_id, msg_body, is_body_string=False):
        for sock in self.connection_list:
            self.send_message(sock, msg_id, msg_body, is_body_string)

    def message_all_but_one(self, ignore, msg_id, msg_body, is_body_string=False):
        for sock in self.connection_list:
            if not sock == ignore:
                self.send_message(sock, msg_id, msg_body, is_body_string)

    # Unused, may be used later but unlikely.
    def send_updates(self):
        for msg in self.updates_out:
            self.message_all(msg.id, msg.body)
        self.updates_out.clear()

    def get_connections_n(self):
        return len(self.connection_list)
