class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def _resize_if_needed(self):
        if self.count / self.size > 0.7:
            self._resize(self.size * 2)

    def _resize(self, new_size):
        old_table = self.table
        self.size = new_size
        self.table = [[] for _ in range(new_size)]
        self.count = 0
        for bucket in old_table:
            for key, value in bucket:
                self[key] = value

    def __setitem__(self, key, value):
        self._resize_if_needed()
        hash_key = self._hash(key)
        bucket = self.table[hash_key]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.count += 1

    def __getitem__(self, key):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]

        for k, v in bucket:
            if k == key:
                return v
        raise KeyError(key)

    def __delitem__(self, key):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.count -= 1
                return
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
        return "\n".join(f"{i}: {bucket}" for i, bucket in enumerate(self.table))


# 测试用例
if __name__ == "__main__":
    print("=== 测试 [] 操作符 ===")
    ht = HashTable(size=5)  # 初始大小设为5便于观察扩容

    # 插入数据
    ht["apple"] = 10
    ht["banana"] = 20
    print("插入 apple=10, banana=20 后:")
    print(ht)
    print(f"当前大小: {len(ht)}, 桶数量: {ht.size}")

    # 更新数据
    ht["apple"] = 15
    print("\n更新 apple=15 后:")
    print(ht["apple"])

    # 触发扩容
    ht["orange"] = 30
    ht["pear"] = 40
    ht["grape"] = 33
    print("\n插入 orange=30, pear=40 后 (应触发扩容):")
    print(ht)
    print(f"当前大小: {len(ht)}, 桶数量: {ht.size}")

    # 删除测试
    del ht["banana"]
    print("\n删除 banana 后:")
    print(ht)
    print(f"当前大小: {len(ht)}")

    # 包含测试
    print("\n检查键是否存在:")
    print("'apple' in ht:", "apple" in ht)
    print("'grape' in ht:", "grape" in ht)