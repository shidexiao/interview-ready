import asyncio


async def worker(name, delay):
    print(f"{name} started")
    await asyncio.sleep(delay)
    print(f"{name} finished")
    return name


async def main():
    # 创建任务
    task1 = asyncio.create_task(worker("Task1", 2))
    task2 = asyncio.create_task(worker("Task2", 1))

    # 等待第一个完成的任务
    done, pending = await asyncio.wait(
        {task1, task2},
        return_when=asyncio.FIRST_COMPLETED
    )

    print(f"First completed: {next(iter(done)).result()}")

    # 取消剩余任务
    for task in pending:
        task.cancel()

    # 等待取消完成
    await asyncio.gather(*pending, return_exceptions=True)


asyncio.run(main())