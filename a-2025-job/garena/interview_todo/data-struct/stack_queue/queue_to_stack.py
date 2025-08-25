from collections import deque

class MyStack:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x: int) -> None:
        self.q1.append(x)

    def pop(self) -> int:
        # 把 q1 中前面元素都转移到 q2，只留最后一个
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        res = self.q1.popleft()  # 栈顶元素
        # 交换 q1 和 q2
        self.q1, self.q2 = self.q2, self.q1
        return res

    def top(self) -> int:
        # 和 pop 类似，但不移除最后一个
        while len(self.q1) > 1:
            self.q2.append(self.q1.popleft())
        res = self.q1[0]  # 取出栈顶
        self.q2.append(self.q1.popleft())  # 放回去
        self.q1, self.q2 = self.q2, self.q1
        return res

    def empty(self) -> bool:
        return not self.q1
