from datetime import datetime, time


def get_today_timestamp():
    # 获取当前日期（时间部分设为 00:00:00）
    today = datetime.combine(datetime.today(), time.min)
    # 转换为毫秒时间戳
    return int(today.timestamp() * 1000)


# # 示例
# today_timestamp = get_today_timestamp()
# today_str = datetime.today().strftime("%Y%m%d")
# print(f"今日日期 {today_str} 的毫秒时间戳: {today_timestamp}")
# # 输出示例: 今日日期 20240520 的毫秒时间戳: 1716163200000


