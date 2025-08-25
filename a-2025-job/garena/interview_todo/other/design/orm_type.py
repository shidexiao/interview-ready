# miniorm.py
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple

# ---------- 工具 ----------
'''
“snake_case” 的写法是 小写 + 下划线分隔，像一条蛇在地上蜿蜒的下划线，
所以叫 snake（和 camelCase 的“骆驼峰”相对）。
Python 的 PEP 8 也推荐函数/变量用 snake_case 命名。
'''

def snake(name: str) -> str:
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

# ---------- 字段 ----------
class Field:
    def __init__(self, column: str | None = None, *, primary_key: bool=False,
                 nullable: bool=True, unique: bool=False, default: Any=None) -> None:
        self.column = column
        self.primary_key = primary_key
        self.nullable = nullable
        self.unique = unique
        self.default = default
        self.name: str | None = None  # set by __set_name__

    def __set_name__(self, owner, name):
        self.name = name
        if self.column is None:
            self.column = name

    # 描述符：把实例属性读写落到 __dict__
    def __get__(self, obj, owner):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)

    def __set__(self, obj, value):
        obj.__dict__[self.name] = value

    # SQLite 简化：类型映射 & DDL 片段
    def sql_type(self) -> str: return "TEXT"
    def ddl(self) -> str:
        parts = [self.column, self.sql_type()]
        if self.primary_key: parts.append("PRIMARY KEY")
        if not self.nullable and not self.primary_key: parts.append("NOT NULL")
        if self.unique: parts.append("UNIQUE")
        return " ".join(parts)

class Integer(Field):
    def sql_type(self) -> str: return "INTEGER"

class String(Field):
    def __init__(self, length: int = 255, **kw):
        super().__init__(**kw)
        self.length = length
    def sql_type(self) -> str: return "TEXT"  # SQLite 不强制长度

# ----- ModelMeta（更稳 & 带诊断）-----
class ModelMeta(type):
    def __new__(mcls, name, bases, ns):
        # 先合并父类 __fields__
        fields = {}
        for base in bases:
            base_fields = getattr(base, "__fields__", None)
            if base_fields:
                fields.update(base_fields)

        # 收集当前类的 Field
        for k, v in list(ns.items()):
            if isinstance(v, Field):
                fields[k] = v

        cls = super().__new__(mcls, name, bases, ns)

        # 元数据挂到类上
        setattr(cls, "__fields__", fields)
        setattr(cls, "__table__", ns.get("__table__") or snake(name))

        # 仅对“具体模型类”做主键校验（跳过基类）
        if name != "Model":
            pks = [f for f in fields.values() if getattr(f, "primary_key", False)]
            # —— 失败就尽早报错（包含诊断信息）
            if len(pks) != 1:
                detail = ", ".join(
                    f"{fname}[*]" if f.primary_key else fname
                    for fname, f in fields.items()
                ) or "(no fields)"
                raise TypeError(
                    f"{name}: expect exactly ONE primary key; got {len(pks)}. "
                    f"fields = {detail}"
                )
            setattr(cls, "__pk__", pks[0])

        return cls
'''
你可以把 ModelMeta.__new__ 理解成“在类创建这一刻，把你在 class 体里写的字段声明（id = Integer(...) 这类）收集成元数据，并安装到类上，顺便做校验与默认值填充”。
这样后面的 create_table / save / get / delete 才知道有哪些列、表名是什么、主键是谁。

元类的作用就是把这段声明，在“类创建那一刻”编译成可用的元数据：
create_table() 需要 __fields__ 和 __table__ 拼 DDL；
save() 要靠 __pk__ 决定 INSERT/UPDATE、拼 WHERE；
get/filter/all() 要知道列名顺序，才能把行数据回填成模型对象。
没有 ModelMeta.__new__ 的收集与安装，这些方法根本不知道类有哪些字段/表名/主键，ORM 就“失明”了。
'''


# ---------- Model 基类 ----------
class Model(metaclass=ModelMeta):
    __fields__: Dict[str, Field]
    __table__: str
    __pk__: Field

    _conn: sqlite3.Connection | None = None
    _debug: bool = False

    # 绑定数据库连接（进程级）
    @classmethod
    def bind(cls, conn: sqlite3.Connection, *, debug: bool=False):
        cls._conn = conn
        cls._debug = debug

    @classmethod
    def _execute(cls, sql: str, params: Iterable[Any] = ()):
        if cls._conn is None:
            raise RuntimeError("Call Model.bind(sqlite3.connect(...)) first.")
        if cls._debug:
            print("SQL:", sql, "\n  params:", list(params))
        cur = cls._conn.cursor()
        cur.execute(sql, tuple(params))
        cls._conn.commit()
        return cur

    @classmethod
    def create_table(cls):
        cols = [f.ddl() for f in cls.__fields__.values()]
        sql = f"CREATE TABLE IF NOT EXISTS {cls.__table__} ({', '.join(cols)})"
        cls._execute(sql)

    @classmethod
    def drop_table(cls):
        sql = f"DROP TABLE IF EXISTS {cls.__table__}"
        cls._execute(sql)

    def __init__(self, **kwargs):
        # 给所有字段赋值（未提供用 default/None）
        for name, f in self.__fields__.items():
            val = kwargs.get(name, f.default)
            setattr(self, name, val)

    # ----- Model.save（早失败 & 信息清晰）-----
    def save(self):
        # 早检查：确保 __pk__ 已存在
        pkf = getattr(type(self), "__pk__", None)
        if pkf is None:
            raise RuntimeError(
                f"{type(self).__name__}.__pk__ is not set. "
                f"Did ModelMeta fail to detect a primary key?"
            )

        pk_name, pk_col = pkf.name, pkf.column
        pk_val = getattr(self, pk_name)

        if pk_val is None:  # INSERT
            cols, vals = [], []
            for name, f in self.__fields__.items():
                if f.primary_key:
                    continue
                cols.append(f.column)
                vals.append(getattr(self, name))
            placeholders = ", ".join(["?"] * len(cols)) or "DEFAULT VALUES"
            if cols:
                sql = f"INSERT INTO {self.__table__} ({', '.join(cols)}) VALUES ({placeholders})"
            else:
                sql = f"INSERT INTO {self.__table__} DEFAULT VALUES"
            cur = self._execute(sql, vals)
            # 自增主键回填
            if isinstance(pkf, Integer):
                setattr(self, pk_name, cur.lastrowid)
        else:  # UPDATE
            sets, vals = [], []
            for name, f in self.__fields__.items():
                if f.primary_key:
                    continue
                sets.append(f"{f.column} = ?")
                vals.append(getattr(self, name))
            vals.append(pk_val)
            sql = f"UPDATE {self.__table__} SET {', '.join(sets)} WHERE {pk_col} = ?"
            self._execute(sql, vals)
        return self

    def delete(self):
        pkf = getattr(type(self), "__pk__", None)  # 从类拿 Field 对象
        if pkf is None:
            raise RuntimeError(f"{type(self).__name__}.__pk__ is not set")

        pk_name, pk_col = pkf.name, pkf.column
        pk_val = getattr(self, pk_name)
        if pk_val is None:
            raise ValueError("Cannot delete a row without primary key value")
        sql = f"DELETE FROM {self.__table__} WHERE {pk_col} = ?"
        self._execute(sql, (pk_val,))

    # ---- 查询工具 ----
    @classmethod
    def _build_where(cls, **kw) -> Tuple[str, List[Any]]:
        if not kw:
            return "", []
        parts = []
        params: List[Any] = []
        for name, val in kw.items():
            f = cls.__fields__.get(name)
            if not f:
                raise AttributeError(f"Unknown field: {name}")
            parts.append(f"{f.column} = ?")
            params.append(val)
        return "WHERE " + " AND ".join(parts), params

    @classmethod
    def _rows_to_models(cls, rows):
        out = []
        names = list(cls.__fields__.keys())
        for r in rows:
            obj = cls()
            for name, val in zip(names, r):
                setattr(obj, name, val)
            out.append(obj)
        return out

    @classmethod
    def get(cls, **kw):
        where, params = cls._build_where(**kw)
        cols = ", ".join(f.column for f in cls.__fields__.values())
        sql = f"SELECT {cols} FROM {cls.__table__} {where} LIMIT 1"
        cur = cls._execute(sql, params)
        row = cur.fetchone()
        if not row:
            return None
        return cls._rows_to_models([row])[0]

    @classmethod
    def filter(cls, **kw):
        where, params = cls._build_where(**kw)
        cols = ", ".join(f.column for f in cls.__fields__.values())
        sql = f"SELECT {cols} FROM {cls.__table__} {where}"
        cur = cls._execute(sql, params)
        return cls._rows_to_models(cur.fetchall())

    @classmethod
    def all(cls):
        cols = ", ".join(f.column for f in cls.__fields__.values())
        sql = f"SELECT {cols} FROM {cls.__table__}"
        cur = cls._execute(sql)
        return cls._rows_to_models(cur.fetchall())

    # 便于打印
    def __repr__(self):
        fields = ", ".join(f"{k}={getattr(self,k)!r}" for k in self.__fields__.keys())
        return f"<{self.__class__.__name__} {fields}>"

# ---------- 示例模型 ----------
class User(Model):
    __table__ = "users"          # 不写也行，会用 snake_case: user
    id = Integer(primary_key=True)    # INTEGER PRIMARY KEY 自增
    name = String(nullable=False)
    age = Integer()



'''
ORM 的元信息是类级别的、应在类创建时确定；
元类 __new__ 是最早、最统一、可校验的落点。
__init_subclass__ 可做简化版；实例化时做会引入时序/并发/性能/语义上的一堆坑。

超简洁版先说结论：
把逻辑放到元类的 __new__ 是为了在**“类创建当下”就把字段/表名/主键信息（元数据）收集+校验+安装到类上，这样不需要任何实例就能调用 User.create_table()、User.get() 等类方法；也能尽早失败**（主键缺失直接在导入时报错），且每个子类都统一生效。

放到实例化时（User().__init__）做不合适：
    需要先 new 一个实例才能有元数据——但 ORM 的 DDL/查询是类级行为；
    会让“第一次实例化”产生副作用（偷偷修改类），时机不稳定、并发有坑；
    性能差（每次实例化都要判定/构建/加锁）；
    很多检查（如“必须恰好一个主键”）就拖到运行期才暴雷；
    描述符/命名空间的钩子（如 __set_name__、__prepare__）只有类创建期能参与。


当类体执行完毕后，Python 用确定的元类调用 metaclass.__new__(mcls, name, bases, namespace, **kw) 来创建类对象，这就是“跳进去”的时机。

参数从哪里来？
name：类名字面量；
bases：类头部的基类元组；
namespace：执行类体得到的字典（包含 __module__、__qualname__、__annotations__、以及赋值的属性）；
**kw：类头部里除 metaclass 之外的其它关键字（若写了会原样转发到 __prepare__/__new__/__init__）。
__set_name__ 在哪里触发？
在 type.__new__ 内部、类对象创建时，对类体里是描述符的属性自动调用（你的 Field 就靠这个拿到 owner 和 name）。

这样你就可以把 ModelMeta.__new__ 里看到的每个值，对应回“类体里写的东西”了。


传给 ModelMeta.__new__ 的参数长什么样？

以 Model 为例（它没有基类）：
name: 'Model'
bases: ()
namespace（关键条目）大致是：
{
  '__module__': 'miniorm',              # 你的模块名
  '__qualname__': 'Model',
  '__annotations__': {
      '__fields__': Dict[str, Field],
      '__table__': str,
      '__pk__': Field
  },
  '_conn': None,
  '_debug': False,
  # 可能还有 '__doc__': None（若无文档字符串）
}
**kw: {}（没有其它关键字参数）

以 User(Model) 为例（它有基类 Model，而且定义了 Field 成员）：
name: 'User'
bases: (Model,)
namespace：
{
  '__module__': 'miniorm',
  '__qualname__': 'User',
  '__table__': 'users',          # 你显式写的
  'id': Integer(primary_key=True),
  'name': String(nullable=False),
  'age': Integer(),
  # 若写了注解也会出现在 '__annotations__'
}


'''



# ---------- 运行示例 ----------
if __name__ == "__main__":
    conn = sqlite3.connect(":memory:")
    Model.bind(conn, debug=True)

    User.drop_table()
    User.create_table()

    # INSERT
    u = User(name="Alice", age=20).save()
    print("after insert:", u)

    # UPDATE
    u.age = 21
    u.save()
    print("after update:", u)

    # GET / FILTER / ALL
    print("get by id:", User.get(id=u.id))
    print("filter age=21:", User.filter(age=21))
    print("all:", User.all())

    # DELETE
    u.delete()
    print("after delete:", User.all())

'''
1) 模块加载阶段（自上而下执行定义）

1, 解释器读取文件，依次执行顶层代码：
定义 snake() 
→ 定义 Field/Integer/String 
→ 定义 元类 ModelMeta 
→ 定义 Model（用 ModelMeta 作为 metaclass）。

2,创建 Model 类时（class Model(metaclass=ModelMeta)）：
调用 ModelMeta.__new__(mcls, name='Model', bases=(), ns=...)
    合并父类字段（无）→ 收集本类字段（无）→ 设 Model.__fields__ = {}、Model.__table__ = 'model'（默认 snake）
    因为 name == 'Model'，跳过主键校验（不设 __pk__）。
返回 Model 类对象（ModelMeta.__init__ 未定义，走默认）。

3, 创建 User 类（class User(Model): ...）：
先执行 User 的类体，把 id/name/age 三个 Field 实例放入类命名空间 ns，再交给元类。
调用 ModelMeta.__new__(..., name='User', bases=(Model,), ns=...)
    合并父类 Model.__fields__（仍为空）。
    收集当前类中的 Field：{'id': Field(...), 'name': Field(...), 'age': Field(...)}。
    创建类对象 cls = type.__new__(...)。
    设 User.__fields__、User.__table__ = "users"（你显式写了 __table__），并找主键：__pk__ = id。
类创建完成后，Python 会自动调用每个描述符的 __set_name__(owner, name)：
    对 id/name/age 依次调用，填充 field.name，若 field.column is None 则设为同名列。

此时，ORM 所需的元数据（__fields__/__table__/__pk__ 等）已挂在 User 类上。



'''