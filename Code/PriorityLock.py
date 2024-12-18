import threading


class PriorityLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.priority_condition = threading.Condition(self.lock)
        self.priority_thread_active = False

    def acquire_priority(self):
        with self.lock:
            while self.priority_thread_active:
                self.priority_condition.wait()
            self.priority_thread_active = True

    def release_priority(self):
        with self.lock:
            self.priority_thread_active = False
            self.priority_condition.notify_all()

    def acquire_non_priority(self):
        with self.lock:
            while self.priority_thread_active:
                self.priority_condition.wait()

    def release_non_priority(self):
        with self.lock:
            self.priority_condition.notify_all()
