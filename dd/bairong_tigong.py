import base64
import hashlib
import json
import urllib.parse
import requests
from datetime import datetime
from enum import Enum
from Crypto.Cipher import AES

block_size = AES.block_size


# 忽略证书校验后可启用此项禁用控制台警告
# requests.packages.urllib3.disable_warnings()


# class
def PKCS5Padding(plain_text):
    number_of_bytes_to_pad = block_size - len(plain_text.encode("utf-8")) % block_size  # python3
    # number_of_bytes_to_pad = block_size - len(plain_text) % block_size #python2
    ascii_string = chr(number_of_bytes_to_pad)
    padding_str = number_of_bytes_to_pad * ascii_string
    padded_plain_text = plain_text + padding_str
    return padded_plain_text.encode("utf-8")  # python3
    # return padded_plain_text #python2


def SHA1PRNG(key):
    signature = hashlib.sha1(key.encode()).digest()
    signature = hashlib.sha1(signature).digest()
    return bytes.fromhex(''.join(['%02x' % i for i in signature])[:32])  # python3
    # return (signature.encode("hex")[:32]).decode("hex")  # python2


def EncodeBase64URLSafeString(result):
    return base64.urlsafe_b64encode(result).decode("utf-8").rstrip("=")  # python3
    # return base64.urlsafe_b64encode(result).rstrip("=") #python2


def AESEncrypt(content, key):
    content = PKCS5Padding(content)
    genKey = SHA1PRNG(key)
    cipher = AES.new(genKey, AES.MODE_ECB)
    return EncodeBase64URLSafeString(cipher.encrypt(content))


def Md5(key):
    return hashlib.md5(key.encode('utf-8')).hexdigest()


class Url(Enum):
    # 生产环境贷前接口地址
    strategyUrl = "https://api2.100credit.cn/strategy_api/v3/hx_query"
    # 生产环境验证接口地址
    verificationUrl = "https://api2.100credit.cn/infoverify/v3/info_verify"
    # 沙箱环境贷前接口地址
    sandboxStrategyUrl = "https://sandbox-api2.100credit.cn/strategy_api/v3/hx_query"
    # 沙箱环境验证接口地址
    sandboxVerificationUrl = "https://sandbox-api2.100credit.cn/infoverify/v3/info_verify"


def riskStrategy():
    # apiCode
    apiCode = "3032705"
    # appKey
    appKey = "b8366b8320efc1642803aeda14e3fe152c344da0e83192106f04d53e0f8c1892"
    # 请求地址
    url = Url.strategyUrl.value
    # 请求参数
    jsonData = {
        # 策略编号 贷前：strategy_id:STRXXXXXX 验证：conf_id:DTVXXXXXX 一次只传其中一个
        # "conf_id": "XXXXXX",
        "strategy_id": "STR_BR0006635",
        # 'strategy_id': 'STR_BR0007563',
        # 身份证号
        "id": '7f4b5e5492a39b80e7a0d51f0025e296',
        # 手机号
        "cell": "b705a9ed98b145adebe022bcbd52aa7c",
        # 姓名
        "name": ""
    }
    print(jsonData)
    checkCode = Md5((json.dumps(jsonData) + apiCode + appKey))
    jsonDataAES = AESEncrypt(urllib.parse.quote(json.dumps(jsonData)), appKey)
    appKeyMD5 = Md5(appKey)

    params = {
        "jsonData": jsonDataAES,
        "apiCode": apiCode,
        "appKey": appKeyMD5,
        "checkCode": checkCode
    }

    res = requests.post(
        # 请求的接口地址
        url,
        # 参数列表
        data=json.dumps(params),
        # 忽略证书校验，开启后不用配置证书
        verify=False,
        # 开启证书校验，该项填证书文件地址
        # verify='/mount_info/BRCA.crt'
    )

    return res.text


st = datetime.now()
res = riskStrategy()
print(datetime.now() - st)
#  汪玉磊
#  15055952004            010ad550351464a10004ee5f149db7d3
#  342423199005291511     39ae8b804d44f2f6f7df70a80057f5cb
