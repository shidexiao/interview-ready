'''

2. MRO 的规则（C3 线性化算法）
Python 使用 C3 线性化算法（C3 Linearization）计算 MRO，规则如下：

子类优先于父类（如 class C(A, B)，则 C 的 MRO 先于 A 和 B）。
继承顺序从左到右（class C(A, B) 的 MRO 会先检查 A，再检查 B）。
单调性：如果 A 在 B 之前出现在 MRO 中，那么 A 的所有父类也会在 B 之前。



三、总结
多继承的正确使用方式
优先使用 Mixin（避免复杂的继承链）。
避免菱形继承（如果必须用，确保 super() 调用正确）。
理解 MRO（ClassName.__mro__ 和 super() 的行为）。

面试常见问题
“Python 如何解决多继承的菱形问题？”（C3 线性化算法）
“Mixin 和普通继承的区别？”（Mixin 不单独实例化，仅提供功能）
“什么时候应该用多继承？什么时候应该用组合？”（组合优于继承，但 Mixin 例外）

如果面试官考察多继承，通常希望候选人：
理解 MRO 机制（super() 不是简单的“调用父类”）。
能合理使用 Mixin（而非滥用多继承）。
能识别设计问题（如 MRO 冲突）。

建议在面试中结合 实际项目经验 回答，例如：
“我在某个项目里用 LoggingMixin 和 CacheMixin 来避免重复代码，同时确保 MRO 顺序正确。”



Python 多继承的应用场景与考察点
多继承（Multiple Inheritance）是 Python 的一个重要特性，但它的使用需要谨慎，否则容易导致代码复杂性和维护性问题。以下是 多继承的典型应用场景 和 面试中的考察点。

一、多继承的应用场景

1. Mixin 模式（核心应用）
场景：

需要为多个类动态添加相同功能，但不想用组合或装饰器。
例如：日志记录、缓存、权限检查等通用功能。

class LoggingMixin:
    def log(self, message):
        print(f"[LOG] {message}")

class DatabaseMixin:
    def save(self):
        print("Data saved to DB")

class User(LoggingMixin, DatabaseMixin):
    def __init__(self, name):
        self.name = name

user = User("Alice")
user.log("User created")  # [LOG] User created
user.save()  # Data saved to DB

考察点：

是否理解 Mixin 的作用（提供功能，但不影响主类逻辑）。
能否区分 Mixin 和普通继承的区别（Mixin 通常不单独实例化）。

2. 接口适配（Interface Adaptation）
场景：

让一个类兼容多个接口（类似 Java 的 implements）。
例如：让一个类同时支持 JSONSerializable 和 DatabaseStorable。

3. 代码复用（谨慎使用）
场景：

多个类有部分相同逻辑，但无法用单一继承表达。
例如：游戏中的 Monster 可能同时继承 Flying 和 FireBreathing。


二、多继承的面试考察点

1. 方法解析顺序（MRO, Method Resolution Order）
class A:
    def foo(self):
        print("A.foo()")

class B(A):
    def foo(self):
        print("B.foo()")

class C(A):
    def foo(self):
        print("C.foo()")

class D(B, C):
    pass

d = D()
d.foo()  # 输出什么？B.foo()
D.__mro__（D -> B -> C -> A）。
是否知道 super() 是按 MRO 顺序调用的。

考察点：

是否理解 super() 是按 MRO 顺序调用的，而非直接调用父类。
能否解释 D.__mro__ 的顺序（D -> B -> C -> A）。
'''

class A:
    def show(self):
        print("A.show()")

class B(A):
    def show(self):
        print("B.show()")

class C(A):
    def show(self):
        print("C.show()")

class D(B, C):
    pass

d = D()
d.show()  # 输出什么？B.show()
print(D.__mro__)
# (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)

