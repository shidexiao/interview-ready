'''
给定一个字符串 s，请你找出其中不含重复字符的最长子串的长度。

输入: s = "abcabcbb"
输出: 3  # 最长无重复子串是 "abc"

输入: s = "bbbbb"
输出: 1  # 最长无重复子串是 "b"

输入: s = "pwwkew"
输出: 3  # 最长是 "wke"


解题思路（滑动窗口 + 哈希表）
核心思想：
使用一个滑动窗口 [left, right] 来表示当前无重复字符的子串。
使用一个 set 或 dict 来记录窗口内字符是否重复。

向右滑动窗口时：
如果遇到重复字符，就移动左指针 left，直到窗口中不再有重复。
更新最大长度。

✅ 滑动窗口适合场景：
查找最长/最短满足某条件的子串/子数组时非常常用。

'''

from typing import Set

def length_of_longest_substring(s: str) -> int:
    seen: Set[str] = set()
    left: int = 0
    max_len: int = 0

    for right in range(len(s)):
        while s[right] in seen: #"abba"这种情况
            seen.remove(s[left])
            left += 1
        seen.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len


