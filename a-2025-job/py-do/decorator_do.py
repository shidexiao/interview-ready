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

if __name__ == '__main__':
    func = greet
    print(func())
    print(great2())
    print(greet3("hello"))
    print('=====')
    print(say_hello.__name__)  # 输出 "say_hello"（而不是 "wrapper"）
    print(say_hello.__doc__)  # 输出 "Original function"
