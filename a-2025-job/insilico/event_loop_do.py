# ======== 示例：两个协程 + 定时器 + 伪IO ========
# 为了简单，这里伪造一个“可写”FD：用 pair 里的一端作为可写信号。
import socket

async def worker_a(loop, sock):
    print("A: start")
    await loop.sleep(0.5)          # -> 放入定时器堆，0.5秒后回到就绪队列
    print("A: after sleep, wait writable")
    await WriteWait(sock)          # -> 注册到 selector，等待可写
    print("A: sock writable, done")

async def worker_b(loop, sock_pair):
    print("B: start")
    await loop.sleep(1.0)
    print("B: make the other side writable")
    # 通过往另一端写点东西，触发我们等待的可写/可读事件
    sock_pair.send(b"ok")          # 让对端收到事件
    print("B: done")

def main():
    loop = EventLoop()

    # 准备一对相连的 socket（本地管道），用来演示 IO 就绪事件
    s1, s2 = socket.socketpair()
    s1.setblocking(False)
    s2.setblocking(False)

    loop.create_task(worker_a(loop, s1))      # 等 s1 可写
    loop.create_task(worker_b(loop, s2))      # 1 秒后写入，让 s1 侧产生事件
    loop.run_forever()

# main()
'''
运行节拍（口头版）

create_task 把两个任务塞进就绪队列

循环推进 worker_a，遇到 await Sleep(0.5) → 挂到定时器堆

推进 worker_b，遇到 await Sleep(1.0) → 挂到定时器堆

事件循环 select(timeout) 等待最早的 0.5 秒到期

0.5 秒到：worker_a 回到就绪队列 → 继续跑，遇到 await WriteWait(s1) → 注册 s1 可写事件到 selector

继续等到 1.0 秒：worker_b 回到就绪队列，继续跑，send(b"ok") 让 s1 对端产生就绪

selector 返回 s1 就绪 → 事件循环把 worker_a 放回就绪队列 → worker_a 继续执行到结束


'''