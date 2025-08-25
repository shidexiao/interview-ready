class HashTableOpenAddressing:
    # 特殊标记，表示已删除的位置
    DELETED = object()

    def __init__(self, size=10):
        self.size = size
        self.table = [None] * size
        self.count = 0  # 记录有效元素数量

    def _hash(self, key, attempt=0):
        """哈希函数 + 线性探测"""
        return (hash(key) + attempt) % self.size

    def _resize_if_needed(self):
        """检查是否需要扩容（负载因子 > 0.7）"""
        if self.count / self.size > 0.7:
            self._resize(self.size * 2)

    def _resize(self, new_size):
        """执行扩容操作"""
        old_table = self.table
        self.size = new_size
        self.table = [None] * new_size
        self.count = 0

        for item in old_table:
            if item is not None and item is not self.DELETED:
                key, value = item
                self[key] = value  # 重新插入

    def __setitem__(self, key, value):
        """插入/更新键值对"""
        self._resize_if_needed()

        attempt = 0
        while True:
            hash_key = self._hash(key, attempt)

            # 情况1：找到空位或已删除的位置
            if self.table[hash_key] is None or self.table[hash_key] is self.DELETED:
                self.table[hash_key] = (key, value)
                self.count += 1
                return

            # 情况2：键已存在，更新值
            existing_key, _ = self.table[hash_key]
            if existing_key == key:
                self.table[hash_key] = (key, value)
                return

            attempt += 1
            if attempt >= self.size:  # 防止无限循环
                raise RuntimeError("哈希表已满")

    def __getitem__(self, key):
        """获取键对应的值"""
        attempt = 0
        while True:
            hash_key = self._hash(key, attempt)
            item = self.table[hash_key]

            if item is None:
                break  # 键不存在

            if item is not self.DELETED:
                existing_key, value = item
                if existing_key == key:
                    return value

            attempt += 1
            if attempt >= self.size:
                break

        raise KeyError(key)

    def __delitem__(self, key):
        """删除键值对"""
        attempt = 0
        while True:
            hash_key = self._hash(key, attempt)
            item = self.table[hash_key]

            if item is None:
                break  # 键不存在

            if item is not self.DELETED:
                existing_key, _ = item
                if existing_key == key:
                    self.table[hash_key] = self.DELETED
                    self.count -= 1
                    return

            attempt += 1
            if attempt >= self.size:
                break

        raise KeyError(key)

    def __len__(self):
        return self.count

    def __contains__(self, key):
        try:
            _ = self[key]
            return True
        except KeyError:
            return False

    def __str__(self):
        return "\n".join(f"{i}: {item}" for i, item in enumerate(self.table))


# 测试用例
if __name__ == "__main__":
    print("=== 开放寻址法测试 ===")
    ht = HashTableOpenAddressing(size=5)

    # 插入数据
    ht["apple"] = 10
    ht["banana"] = 20
    print("插入 apple=10, banana=20 后:")
    print(ht)

    # 冲突测试（故意选择会冲突的键）
    ht["orange"] = 30  # 假设与apple冲突
    print("\n插入 orange=30 后（模拟冲突）:")
    print(ht)

    # 删除测试
    del ht["apple"]
    print("\n删除 apple 后:")
    print(ht)

    # 读取测试
    print("\n读取 banana 的值:", ht["banana"])
    try:
        print(ht["apple"])
    except KeyError as e:
        print("读取已删除的键:", e)
