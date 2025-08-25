import random
import time
from confluent_kafka import Producer

services = ['payment', 'order', 'inventory']
levels = ['INFO', 'WARN', 'ERROR']

producer = Producer({'bootstrap.servers': '121.40.196.238:9092'})

while True:
    service = random.choice(services)
    level = random.choice(levels)
    log_msg = f"{time.ctime()} [{level}] {service}: Sample log message"
    producer.produce('app_logs', value=log_msg.encode('utf-8'))
    producer.flush()
    time.sleep(0.5)