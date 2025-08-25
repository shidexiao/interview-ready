# 示例：有效的括号
def isValid(s):
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack


def isValid2(s: str) -> bool:
    stack = []
    mapping = {'(': ')', '[': ']', '{': '}'}
    for ch in s:
        if ch in mapping:
            stack.append(mapping[ch])   # 直接压入“期望看到的右括号”
        else:
            if not stack or stack.pop() != ch:
                return False
    return not stack


if __name__ == '__main__':
    s = "()[]{}"
    print(isValid(s))

    '''
    字符串：([{}])
操作过程：
( → 入栈 [)]
[ → 入栈 [), ]
{ → 入栈 [), ], }
} → 出栈比对 → 匹配 ✅
] → 出栈比对 → 匹配 ✅
) → 出栈比对 → 匹配 ✅
扫描完毕，栈空 → 有效
    '''