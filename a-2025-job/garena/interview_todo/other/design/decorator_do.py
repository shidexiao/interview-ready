from functools import wraps
'''

Python 装饰器的底层原理
装饰器的本质是 高阶函数 + 闭包 + 函数对象 的结合，其核心原理如下：
1,函数是一等对象（First-Class Object）
Python 中函数是对象，可以被赋值、作为参数传递、作为返回值返回。
2, 闭包（Closure）
装饰器利用闭包保存外层函数的变量（如被装饰的函数 func），即使外层函数已执行完毕。
3, 语法糖 @ 的等价形式
装饰器 @decorator 本质是函数调用的语法糖：
4, 装饰器的执行时机
装饰器在 模块加载时立即执行（即函数定义时），而非函数调用时：


"装饰器模式与 Python 装饰器有何联系？"
答：Python 装饰器是装饰器模式的语法糖实现，动态扩展对象功能，符合开放-封闭原则（OCP）。



"Python 中函数为什么可以作为参数传递？"
答：函数是一等对象，本质是 function 类的实例，拥有 __call__ 方法。


1. 基础装饰器
2. 保留原函数元信息
3. 线程安全的装饰器
4. 带参数的装饰器
5. 类装饰器
6. 方法装饰器（用于类方法）
7. 装饰器堆叠（多个装饰器）
8. 异步装饰器（用于 async/await）
9. 属性装饰器（@property）
10. 类装饰器（装饰类）


'''
def greet():
    return "hello"

def outer(func):
    print('decorator executed')
    def inner(*args, **kwargs):
        return func(*args, **kwargs).upper()
    return inner

@outer
def great2():
    return "hello"


def basic_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"After calling {func.__name__}")
        return result
    return wrapper

@basic_decorator
def greet3(name):
    print(f"Hello, {name}!")
    return 'greet3'


def preserve_metadata(func):
    @wraps(func)  # 保留原函数信息
    def wrapper(*args, **kwargs):
        """Wrapper function"""
        return func(*args, **kwargs)
    return wrapper

@preserve_metadata
def say_hello():
    """Original function"""
    print("Hello!")


import threading

def thread_safe(func):
    lock = threading.Lock()
    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:  # 加锁
            return func(*args, **kwargs)
    return wrapper

@thread_safe
def update_shared_data():
    # 操作共享资源
    pass


from __future__ import annotations
import time
import random
import threading
import logging
from typing import Callable, TypeVar, ParamSpec, Iterable, Optional
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")


# ========= 1) 重试装饰器（含退避与抖动） =========
def retry(
    exceptions: Iterable[type[BaseException]] = (Exception,),
    tries: int = 3,
    delay: float = 0.2,
    backoff: float = 2.0,
    jitter: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    对抛出指定异常的函数进行重试。
    - exceptions: 需要捕获并重试的异常类型
    - tries: 最大重试次数（包含首次执行，3 就是最多再试 2 次）
    - delay: 初始等待时间（秒）
    - backoff: 指数退避倍率（>1）
    - jitter: 抖动幅度（0~jitter 之间的随机加成，避免惊群）
    """
    if tries < 1:
        raise ValueError("tries must be >= 1")
    if backoff <= 0:
        raise ValueError("backoff must be > 0")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _delay = delay
            attempt = 1
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:  # 只对指定异常重试
                    if attempt >= tries:
                        raise
                    wait = _delay + (random.random() * jitter if jitter > 0 else 0.0)
                    (logger or logging.getLogger(__name__)).warning(
                        "retrying %s after error (%s), attempt %d/%d, sleep %.3fs",
                        func.__name__, e, attempt, tries, wait,
                    )
                    time.sleep(wait)
                    _delay *= backoff
                    attempt += 1
        return wrapper
    return decorator


# ========= 2) 使用 functools.wraps 的标准装饰器示例 =========
def timeit(func: Callable[P, R]) -> Callable[P, R]:
    """
    计时装饰器示例：保留原函数的 __name__ / __doc__ / 注解 等元数据。
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            dt = (time.perf_counter() - t0) * 1000
            print(f"[timeit] {func.__name__} took {dt:.2f} ms")
    return wrapper


# ========= 3) 线程安全装饰器（加锁） =========
# 3.1 全局/函数级锁（适合保护共享资源，所有调用串行）
def synchronized(lock: Optional[threading.RLock] = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    将目标函数用同一个 Lock 保护；不传 lock 时为该函数创建一个独立锁。
    适合函数级的临界区保护（例如更新全局缓存/写文件等）。
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        _lock = lock or threading.RLock()

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with _lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator


# 3.2 每实例锁（适合类方法：不同实例互不影响；同一实例内串行）
def synchronized_method(attr_name: str = "_lock") -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    给“实例方法”加锁：每个实例有自己的 RLock（懒加载）。
    用于保护同一对象内部共享状态（计数器、缓存、文件句柄等）。
    """
    def decorator(method: Callable[P, R]) -> Callable[P, R]:
        @wraps(method)
        def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[no-redef]
            lock = getattr(self, attr_name, None)
            if lock is None:
                lock = threading.RLock()
                setattr(self, attr_name, lock)
            with lock:
                return method(self, *args, **kwargs)
        return wrapper
    return decorator


# ====================== 使用示例 ======================

@retry(exceptions=(ValueError,), tries=5, delay=0.1, backoff=1.5, jitter=0.05)
@timeit
def flaky_op(x: int) -> int:
    """有时会抛 ValueError 的函数，演示重试+计时叠加装饰。"""
    if x < 3:
        raise ValueError("boom")
    return x * 2


shared_lock = threading.RLock()

@synchronized(shared_lock)  # 所有对 write_log 的并发调用都会串行执行
def write_log(line: str) -> None:
    with open("app.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Counter:
    def __init__(self) -> None:
        self.value = 0

    @synchronized_method()  # 同一个实例内的并发自增是安全的；不同实例互不影响
    def inc(self) -> None:
        self.value += 1


if __name__ == '__main__':
    func = greet
    print(func())
    print(great2())
    print(greet3("hello"))
    print('=====')
    print(say_hello.__name__)  # 输出 "say_hello"（而不是 "wrapper"）
    print(say_hello.__doc__)  # 输出 "Original function"
