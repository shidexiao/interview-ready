class MyQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x: int) -> None:
        # 入队：压入输入栈
        self.in_stack.append(x)

    def pop(self) -> int:
        # 出队：先确保 out_stack 有元素
        self._move()
        return self.out_stack.pop()

    def peek(self) -> int:
        # 查看队头
        self._move()
        return self.out_stack[-1]

    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack

    def _move(self) -> None:
        # 如果 out_stack 为空，把 in_stack 元素倒过去
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
