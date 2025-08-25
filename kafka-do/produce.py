from kafka import KafkaProducer
import json
import time

# 创建生产者
producer = KafkaProducer(
    bootstrap_servers=['121.40.196.238:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    # 增加以下参数
    request_timeout_ms=30000,  # 调大超时时间
    retries=5  # 增加重试
)

# 发送消息
for i in range(1000):
    message = {
        'id': i,
        'message': f'Test message {i}',
        'timestamp': int(time.time())
    }
    # 发送到test_topic主题
    producer.send('test_topic', value=message)
    print(f"Sent: {message}")
    time.sleep(1)

# 确保所有消息都已发送
producer.flush()