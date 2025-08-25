
# ======== 基础工具：Future / Awaitables / Task ========

class Future:
    # “一个稍后会有结果的占位符”
    def __init__(self):
        self._done = False
        self._result = None
        self._callbacks = []   # 完成后唤醒谁（一般是 Task）

    def done(self): return self._done
    def result(self): return self._result

    def add_done_callback(self, cb):
        if self._done: cb(self)
        else: self._callbacks.append(cb)

    def set_result(self, value):
        self._done = True
        self._result = value
        # 通知挂在上面的 Task：“条件好了，可以继续推进你了！”
        for cb in self._callbacks: cb(self)
        self._callbacks.clear()

    # 让 Future 可被 `await`：协程 await 它时，会“暂停”并把控制权交回事件循环
    def __await__(self):
        # 约定：yield self 把自己“交给事件循环”，等事件循环把我标记 done 后再恢复
        result = yield self
        return result

class Sleep:
    # 定时器 awaitable：await Sleep(1.5)
    def __init__(self, delay): self.delay = delay
    def __await__(self):
        # 约定：yield ("sleep", delay) 告诉事件循环“把我挂到定时器”
        yield ("sleep", self.delay)
        return None

class ReadWait:
    # IO awaitable：等待 socket 可读
    def __init__(self, sock): self.sock = sock
    def __await__(self):
        # 约定：yield ("read", sock) 告诉事件循环“把我注册进多路复用器等可读”
        yield ("read", self.sock)
        return None

class WriteWait:
    # IO awaitable：等待 socket 可写
    def __init__(self, sock): self.sock = sock
    def __await__(self):
        yield ("write", self.sock)
        return None

class Task:
    """
    Task = 协程的运行容器 + 状态机
    生命周期：PENDING -> RUNNING -> (DONE | CANCELLED)
    """
    PENDING, RUNNING, DONE, CANCELLED = range(4)

    def __init__(self, coro, loop):
        self.coro = coro            # 协程对象（保存了“暂停点”和局部变量）
        self.loop = loop
        self.state = Task.PENDING
        self.result_value = None
        self.exception = None

    def step(self, send_value=None):
        """把协程推进“一小步”：直到下一个 await 处暂停"""
        if self.state in (Task.DONE, Task.CANCELLED): return
        self.state = Task.RUNNING
        try:
            # 协程继续跑，直到再次遇到 await（对应 __await__ 的 yield）
            awaited = self.coro.send(send_value)
        except StopIteration as stop:
            # 协程自然返回 -> DONE
            self.state = Task.DONE
            self.result_value = getattr(stop, "value", None)
            self.loop._task_done(self)   # 让事件循环做善后
            return
        except Exception as e:
            self.state = Task.DONE
            self.exception = e
            self.loop._task_done(self)
            return

        # 根据 await 的对象类型，告诉事件循环“把我挂哪儿去”
        if isinstance(awaited, Future):
            # 绑定回调：Future 一完成 -> 把我（Task）放回就绪队列
            awaited.add_done_callback(lambda fut: self.loop._ready.append(self))
        elif isinstance(awaited, tuple):
            kind, payload = awaited
            if kind == "sleep":
                self.loop._add_timer(self, delay=payload)
            elif kind == "read":
                self.loop._wait_readable(self, sock=payload)
            elif kind == "write":
                self.loop._wait_writable(self, sock=payload)
        else:
            # 为了简化，其他类型忽略
            pass

    # 供事件循环把 Task 重新推进（比如 IO/定时器就绪后）
    def resume(self):
        self.loop._ready.append(self)


