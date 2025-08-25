from confluent_kafka import Consumer

conf = {
    'bootstrap.servers': '121.40.196.238:9092',
    'group.id': 'log_consumer_group_2'  # 另一个唯一组ID
}
consumer = Consumer(conf)
consumer.subscribe(['app_logs'])

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    log = msg.value().decode('utf-8')
    with open('app_consumer2.log', 'a') as f:
        f.write(log + '\n')
    print(f"[Consumer2] Logged: {log}")