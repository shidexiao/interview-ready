import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

class AesException(Exception):
    pass

def encrypt(content, pwd_key):
    if not content or not pwd_key:
        raise ValueError("参数不合法")

    try:
        # 使用SHA-1生成密钥
        # key = hashlib.sha1(pwd_key.encode('utf-8')).digest()[:16]  # 生成16字节的AES密钥
        signature = hashlib.sha1(pwd_key.encode()).digest()
        signature = hashlib.sha1(signature).digest()
        key = bytes.fromhex(''.join(['%02x' % i for i in signature])[:32])

        # 初始化AES加密器
        cipher = AES.new(key, AES.MODE_ECB)

        # 加密并使用Base64编码
        byte_content = content.encode('utf-8')
        padded_content = pad(byte_content, AES.block_size)
        encrypted_bytes = cipher.encrypt(padded_content)
        return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
    except Exception as e:
        raise AesException("AESException: 对参数进行AES加解密过程中异常") from e


if __name__ == '__main__':
    # 示例用法
    try:
        # encrypted_data = encrypt("Hello, World!", "password123")
        encrypted_data = encrypt("Hello, World!", "password123")
        print("Encrypted:", encrypted_data)
    except AesException as e:
        print("Error:", str(e))
