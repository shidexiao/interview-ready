'''
元类（Metaclass）是 Python 中最高级的特性之一，它允许开发者 控制类的创建行为。
虽然日常开发中很少直接使用，
但在框架设计（如 Django ORM、SQLAlchemy）、API 约束、动态代码生成等场景中非常有用。
以下是元类的核心应用场景和面试考察点。

一、元类的应用场景
1. 动态修改或增强类的行为
场景：
在类定义时自动添加方法、属性或修改继承关系。
例如：自动给所有方法添加日志、性能统计、权限检查等装饰器。

2. 强制接口约束（类似抽象基类 ABC）
场景：
确保子类必须实现某些方法（类似 Java 的 interface）。
例如：要求所有子类必须实现 save() 方法。

3. ORM 框架（如 Django、SQLAlchemy）
场景：
动态生成数据库表模型（如 Django 的 Model）。
例如：class User(models.Model) 自动映射到数据库表。


二、元类的面试考察点

1. 元类的基本概念
问题：
“什么是元类？它和类的关系是什么？”
“type 和 object 的关系是什么？”
答案：
元类是类的类，控制类的创建（class 关键字背后的机制）。
type 是默认的元类，所有类（包括 object）都是 type 的实例。
2. __new__ vs __init__ 在元类中的区别
考察点：
__new__ 负责创建类对象（返回类）。
__init__ 负责初始化类（不能返回类）。

3. 动态创建类
问题：
“如何用 type() 动态创建类？”
def say_hello(self):
    print(f"Hello, {self.name}")

User = type("User", (), {"name": "Alice", "say_hello": say_hello})
user = User()
user.say_hello()  # Hello, Alice
是否理解 type(name, bases, attrs) 的作用？

4. 元类的继承问题
是否知道多继承时元类必须兼容？

三、总结
元类的核心用途
动态修改类（如自动装饰方法）。
强制接口约束（类似抽象基类）。
ORM 框架（如 Django 的 Model）。
单例模式（全局唯一实例）。

面试高频问题
“元类和类装饰器有什么区别？”（元类作用于类定义阶段，装饰器作用于类创建后）
“Django ORM 如何用元类实现？”（通过 ModelBase 动态生成数据库字段）
“什么时候应该用元类？”（框架设计、API 约束，普通业务代码尽量避免）

回答技巧
结合具体项目经验（如：“我用元类实现了一个自动注册所有子类的插件系统”）。
强调 元类的复杂性，避免滥用（“除非必要，否则优先用类装饰器”）。
元类是 Python 最强大的特性之一，但也是最容易被误用的。面试时重点考察 设计思想 而非死记硬背语法。
'''

class LoggingMeta(type):
    def __new__(cls, name, bases, attrs):
        # 遍历所有属性，如果是函数，就包装它
        for attr_name, attr_value in attrs.items():
            if callable(attr_value):
                attrs[attr_name] = cls.log_method(attr_value)
        return super().__new__(cls, name, bases, attrs)

    @staticmethod
    def log_method(method):
        def wrapper(*args, **kwargs):
            print(f"[LOG] Calling {method.__name__}")
            return method(*args, **kwargs)
        return wrapper

class User(metaclass=LoggingMeta):
    def login(self):
        print("User logged in")

user = User()
user.login()  # 输出: [LOG] Calling login \n User logged in
    


####ORM
class ModelMeta(type):
    def __new__(cls, name, bases, attrs):
        # 收集所有字段属性
        fields = {}
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, Field):
                fields[attr_name] = attr_value
        attrs["_fields"] = fields
        return super().__new__(cls, name, bases, attrs)

class Field:
    def __init__(self, type_):
        self.type = type_

class Model(metaclass=ModelMeta):
    pass

class User(Model):
    name = Field(str)
    age = Field(int)

print(User._fields)  # {'name': <Field object>, 'age': <Field object>}