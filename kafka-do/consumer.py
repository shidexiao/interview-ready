from kafka import KafkaConsumer
import json

# 创建消费者
consumer = KafkaConsumer(
    'test_topic',
    bootstrap_servers=['121.40.196.238:9092'],
    auto_offset_reset='earliest',  # 从最早的消息开始读取
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    group_id='test_group'
)

# 消费消息
print("Waiting for messages...")
for message in consumer:
    print(f"Received: {message.value}")