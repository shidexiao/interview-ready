
# ======== 事件循环（全局调度者）========

import selectors, heapq, time
from collections import deque

class EventLoop:
    def __init__(self):
        self.selector = selectors.DefaultSelector()  # IO 多路复用器（底层 epoll/kqueue/IOCP）
        self._ready = deque()                        # 就绪队列：马上可推进的 Task
        self._timers = []                            # 小根堆：(deadline, seq, task)
        self._seq = 0                                # 保证堆稳定性

    # ---- 调度 API ----
    def create_task(self, coro) -> Task:
        t = Task(coro, self)
        self._ready.append(t)        # 新任务一创建 -> 进入就绪队列
        return t

    async def sleep(self, delay):
        return await Sleep(delay)

    # ---- 内部：定时器/IO 等待 ----
    def _add_timer(self, task: Task, delay: float):
        heapq.heappush(self._timers, (time.time()+delay, self._seq, task))
        self._seq += 1

    def _wait_readable(self, task: Task, sock):
        self.selector.register(sock, selectors.EVENT_READ, task)

    def _wait_writable(self, task: Task, sock):
        self.selector.register(sock, selectors.EVENT_WRITE, task)

    # ---- 事件循环主线 ----
    def run_forever(self):
        while True:
            # 1) 先处理到期的定时器 -> 放回就绪队列
            now = time.time()
            while self._timers and self._timers[0][0] <= now:
                _,_,task = heapq.heappop(self._timers)
                self._ready.append(task)

            # 2) 推进所有“当前就绪”的 Task（每个只推进一小步）
            n_ready = len(self._ready)
            for _ in range(n_ready):
                task = self._ready.popleft()
                task.step()   # 可能再次挂起（sleep/IO），也可能结束（StopIteration）

            # 3) 计算 selector 的超时（最近的定时器到期时间 - now）
            timeout = None
            if self._timers:
                timeout = max(0.0, self._timers[0][0] - time.time())
            if self._ready:
                timeout = 0.0   # 还有就绪任务，就别等了

            # 4) 等待 IO 事件（或超时唤醒以处理下一个定时器）
            events = self.selector.select(timeout)
            for key, _ in events:
                task = key.data           # 我们把 task 塞在 data 里
                self.selector.unregister(key.fileobj)
                self._ready.append(task)  # 对应 FD 就绪 -> 把 Task 放回就绪队列

    # Task 完成后的善后（可扩展回调/链式唤醒等）
    def _task_done(self, task: Task):
        pass

'''
谁在“全局控制”？
事件循环（EventLoop.run_forever）：唯一的“永动机”。

它每一轮都做三件事：
把到时的定时器放回就绪；
推进当前就绪的 Task 各一步；
用多路复用器等待 IO/下一次定时器。

主线是什么？
主线就是事件循环的那条 while True。所有协程都只是被它“推进与挂起”的对象。

怎么调度起来的？
调度点只在协程主动 await 的地方（用户态协作式切换）。
await 的对象告诉事件循环：“我要等 定时器 / 可读 / 可写 / 某个 Future 完成”，循环据此把 Task 挂起并登记事件；事件发生后再把 Task 丢回就绪队列，推进下一步。


'''
