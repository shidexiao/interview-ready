'''

要思路（5 步法）：
统计目标字符串 t 中每个字符需要的次数（用 Counter 存储）

使用双指针 left 和 right 构造一个“窗口”

向右移动 right 指针，把字符加入窗口中，记录每个字符的出现次数

当窗口中字符满足了 t 中所有字符需求（字符数足够），就开始尝试收缩 left 指针，缩短窗口长度

不断更新最小窗口长度，并记录其起始位置

还要记录窗口的大小。
'''

from typing import Dict
from collections import Counter

def min_window(s: str, t: str) -> str:
    if not s or not t:
        return ""

    need: Dict[str, int] = Counter(t)  # 统计 t 中每个字符的需求
    window: Dict[str, int] = {}        # 当前窗口中字符的计数
    have, need_count = 0, len(need)

    left = 0
    min_len = float('inf')
    res = (0, 0)

    for right, char in enumerate(s):
        window[char] = window.get(char, 0) + 1

        # 当前字符满足了需要
        if char in need and window[char] == need[char]:
            have += 1

        # 窗口已经覆盖了所有目标字符
        while have == need_count:
            # 更新最小窗口
            if right - left + 1 < min_len:
                res = (left, right)
                min_len = right - left + 1

            # 尝试收缩窗口
            window[s[left]] -= 1
            if s[left] in need and window[s[left]] < need[s[left]]:
                have -= 1
            left += 1

    l, r = res
    return s[l:r+1] if min_len != float('inf') else ""



