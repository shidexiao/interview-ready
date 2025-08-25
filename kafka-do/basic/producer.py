from confluent_kafka import Producer
import time

conf = {'bootstrap.servers': '121.40.196.238:9092'}
producer = Producer(conf)

topic = 'test_topic'

for i in range(10):
    message = f"Message {i} at {time.strftime('%X')}"
    producer.produce(topic, value=message.encode('utf-8'))
    print(f"Sent: {message}")
    producer.flush()
    time.sleep(1)