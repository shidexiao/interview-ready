def outer(func):
    print('decorator executed')
    def inner(*args, **kwargs):
        return func(*args, **kwargs).upper()
    return inner

def log(text):
    def outer(func):
        print(f'decorator executed {text}')
        def inner(*args, **kwargs):
            return func(*args, **kwargs).upper()
        return inner
    return outer

@outer
def greet1():
    return "hello"

print(greet1.__name__)
@outer
def greet2(name, a='b'):
    return f"hello-{name}-{a}"
print(greet2.__name__)

from functools import wraps
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

print(say_hello.__name__)


if __name__ == '__main__':
    print(greet1())
    print(greet2('xxx'))