from typing import List

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    # 1) 总油量 < 总消耗 → 肯定不可能跑完一圈
    if sum(gas) < sum(cost):
        return -1

    start = 0   # 2) 候选起点
    tank = 0    # 3) 从当前起点累计到当前位置的油量余额
    for i in range(len(gas)):
        tank += gas[i] - cost[i]  # 到第 i 站，先加气再开走的净变化
        if tank < 0:
            # 4) 若余额跌破 0，说明从 'start' 到 i 这段整体是亏的
            #    起点不可能在 [start, i] 任何位置 → 把起点跳到 i+1 继续试
            start = i + 1
            tank = 0
    return start  # 5) 由于总和 >= 0，最终这个 start 就是唯一可行解


class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        i = 0
        while i < n:
            sum_of_gas = sum_of_cost = 0
            cnt = 0
            while cnt < n:
                j = (i + cnt) % n
                sum_of_gas += gas[j]
                sum_of_cost += cost[j]
                if sum_of_cost > sum_of_gas:
                    break
                cnt += 1
            if cnt == n:
                return i
            else:
                i += cnt + 1
        return -1
