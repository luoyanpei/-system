import threading


class Condition:
    def __init__(self):
        self.__signaled = False
        self.__lock = threading.Lock()
        self.__condition = threading.Condition(self.__lock)

    def notify(self):
        self.__lock.acquire()
        self.__signaled = True
        self.__condition.notify()
        self.__lock.release()

    def wait(self, timeout_ms):
        try:
            self.__lock.acquire()
            if not self.__condition.wait_for(lambda: self.__signaled, timeout_ms/1000):
                return False

            self.__signaled = False
            return True
        except ValueError:
            return False
        finally:
            self.__lock.release()
