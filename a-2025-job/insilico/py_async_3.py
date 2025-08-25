import asyncio
from random import random


async def limited_worker(semaphore, name, delay):
    async with semaphore:
        print(f"{name} started")
        await asyncio.sleep(delay)
        print(f"{name} finished")
        return name


async def main():
    # 限制并发数为2
    semaphore = asyncio.Semaphore(2)

    tasks = [
        limited_worker(semaphore, f"Task-{i}", random() * 2)
        for i in range(5)
    ]

    results = await asyncio.gather(*tasks)
    print("All done:", results)


asyncio.run(main())