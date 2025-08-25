class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 测试
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True


def singleton(cls):
    _instances = {}

    def wrapper(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return wrapper

@singleton
class Singleton:
    pass

# 测试
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True


import threading

class Singleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # 双重检查锁定
                    cls._instance = super().__new__(cls)
        return cls._instance

# 测试
def test_singleton():
    s = Singleton()
    print(id(s))

threads = []
for _ in range(10):
    t = threading.Thread(target=test_singleton)
    threads.append(t)
    t.start()

for t in threads:
    t.join()