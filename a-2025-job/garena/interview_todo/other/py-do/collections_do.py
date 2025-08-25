from collections import deque
from collections import Counter


dq = deque([1,2,3])
dq.appendleft(0)
print(dq)
print(dq.pop())
print(dq.popleft())
print('---')
print(dq)
print(dq.append(4))
print(dq.popleft())

print('=======')
cnt = Counter('banana')
print(cnt)
print(cnt.most_common(2))
cnt = Counter(['apple', 'banana', 'orange', 'apple', 'orange', 'banana', 'apple'])
print(cnt)
print(cnt.get('apple'))
print(cnt.most_common(3))
