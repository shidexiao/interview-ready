

'''
反转链表（Reverse Linked List）
给定一个单链表的头节点 head，请反转链表，并返回反转后的头节点。
输入: 1 -> 2 -> 3 -> 4 -> 5 -> None
输出: 5 -> 4 -> 3 -> 2 -> 1 -> None

解题思路：
反转链表的关键是将链表中的每个节点的 next 指针指向前一个节点。

需要使用两个辅助指针：
prev 指向已经反转好的链表部分（初始为 None）
curr 指向当前处理的节点（初始为 head）

遍历链表：
暂存当前节点的下一个节点 next_node = curr.next
将当前节点指向 prev
移动 prev 和 curr 指针往后推进
最终 prev 指向新的头节点
'''

from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    prev: Optional[ListNode] = None
    curr: Optional[ListNode] = head

    while curr:
        next_node = curr.next  # 先保存下一个节点
        curr.next = prev       # 反转指针
        prev = curr            # prev 向前移动
        curr = next_node       # curr 向前移动

    return prev  # prev 是反转后的头节点


