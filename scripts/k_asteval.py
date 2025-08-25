from asteval import Interpreter

'''
https://lmfit.github.io/asteval/
'''
aeval = Interpreter()
# 定义输入项，配置数据库
input_terms_types = {'a':'int', 'b': 'float', 'norm': 'string'}
# 定义输入项规则逻辑，配置数据库
relation_code = """(((a + b) > norm['mykey']) or (a > b)) and d is None"""

input_data = {'a': 123, 'b': 32.0, 'norm': {'mykey': None}, 'd': 90}
for p,v in input_terms_types.items():
    aeval.symtable[p] = input_data.get(p)
res = aeval(relation_code)
print(res)

aeval('x = sqrt(3)')
aeval('print(x)')
aeval('''
for i in range(10):
    print(i, sqrt(i), log(1+1))
''')


# TODO 上线校验
# 定义输入项和 规则逻辑是否符合
# 随机input_data 模拟一下验证规则



'''
项目介绍
ASTEVAL 是一款基于 Python 的 ast 模块开发的安全且高效的小型数学语言解析器，专注于处理用户输入的数学表达式。这个库的核心目标是在提供简单、安全和稳定的计算环境的同时，支持多种Python语言结构。

项目技术分析
ASTEVAL 利用了Python的抽象语法树（AST）机制来解析和执行代码，使得它能够支持许多Python的语言特性，如切片、子脚本操作、列表推导式、条件语句、流程控制结构（for循环、while循环、try-except-finally块）等。此外，它还提供了内置数据类型的完全支持，包括字典、元组、列表、NumPy数组和字符串。
如果NumPy模块可用，ASTEVAL会导入并利用其大量的数学函数。为了提高安全性，它限制了一些可能导致潜在风险的功能，例如创建类、导入模块、执行文件、函数装饰器、yield、lambda以及exec和eval等。

项目及技术应用场景
ASTEVAL 非常适合在以下场景中使用：

教育：在线教育平台可以使用ASTEVAL让学习者在安全的环境中尝试和理解Python表达式。
数据分析工具：在数据分析应用中，用户可以通过ASTEVAL快速计算自定义表达式，无需编写完整的Python代码。
命令行工具：需要用户交互式计算的命令行程序，可以集成ASTEVAL作为简单的表达式计算引擎。
配置文件：允许用户通过表达式定制配置的行为。
'''