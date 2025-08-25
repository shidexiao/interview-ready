from confluent_kafka import Consumer

conf = {
    'bootstrap.servers': '121.40.196.238:9092',
    'group.id': 'test_group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(conf)
consumer.subscribe(['test_topic'])

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        print(f"Error: {msg.error()}")
    else:
        print(f"Received: {msg.value().decode('utf-8')}")