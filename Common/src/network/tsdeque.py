from collections import deque
from threading import Lock


class TsDeque:
    def __init__(self):
        self.deque = deque([])
        self.mutex = Lock()

    def empty(self):
        with self.mutex:
            return not bool(self.deque)

    def size(self):
        self.mutex.acquire()
        length = len(self.deque)
        self.mutex.release()
        return length

    def clear(self):
        self.mutex.acquire()
        self.deque.clear()
        self.mutex.release()

    def front(self):
        with self.mutex:
            return self.deque[0]

    def back(self):
        with self.mutex:
            return self.deque[-1]

    def append_left(self, item):
        self.mutex.acquire()
        self.deque.appendleft(item)
        self.mutex.release()

    def append(self, item):
        self.mutex.acquire()
        self.deque.append(item)
        self.mutex.release()

    def pop_left(self):
        with self.mutex:
            first = self.deque[0]
            self.deque.popleft()
            return first

    def pop_right(self):
        with self.mutex:
            last = self.deque[-1]
            self.deque.pop()
            return last
