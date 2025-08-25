import base64

# 假设有二进制数据
binary_data = b'Hello, world!'

# 编码成 Base64
base64_encoded = base64.b64encode(binary_data)
print(base64_encoded)

# 转换成字符串格式（可选）
base64_string = base64_encoded.decode('utf-8')

print(base64_string)  # 输出: SGVsbG8sIHdvcmxkIQ==


with open('score_acard_husui_20250122-2.cdpkl', 'rb') as f:
    binary_data = f.read()

# 编码为 Base64
base64_encoded = base64.b64encode(binary_data)
print(base64_encoded.decode('utf-8'))

