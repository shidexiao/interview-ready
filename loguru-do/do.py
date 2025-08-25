from loguru import logger

logger.add("普通日志.log", filter=lambda x: "[普通]" in x["message"])
logger.add("警告日志.log", filter=lambda x: "[需要注意]" in x["message"])
logger.add("致命错误.log", filter=lambda x: "[致命]" in x["message"])

logger.info("[普通]我是一条普通日志")
logger.warning("[需要注意]xx 写法在下个版本将会移除，请做好迁移")
logger.error("[致命]系统启动失败！")


# 一日一技：loguru 如何把不同的日志写入不同的文件中
# [python] Python日志记录库loguru使用指北  https://www.cnblogs.com/luohenyueji/p/18276299

