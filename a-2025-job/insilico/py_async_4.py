import asyncio
import random


async def producer(queue, name):
    for i in range(3):
        item = f"{name}-{i}"
        await asyncio.sleep(random.random())
        await queue.put(item)
        print(f"Produced {item}")
    await queue.put(None)  # 结束信号


async def consumer(queue, name):
    while True:
        item = await queue.get()
        if item is None:
            queue.put(None)  # 让其他消费者也能结束
            break
        print(f"{name} consumed {item}")
        await asyncio.sleep(random.random())


async def main():
    queue = asyncio.Queue(maxsize=2)

    producers = [
        asyncio.create_task(producer(queue, f"Producer-{i}"))
        for i in range(2)
    ]

    consumers = [
        asyncio.create_task(consumer(queue, f"Consumer-{i}"))
        for i in range(3)
    ]

    await asyncio.gather(*producers)
    await queue.join()
    for c in consumers:
        c.cancel()
    await asyncio.gather(*consumers, return_exceptions=True)


asyncio.run(main())