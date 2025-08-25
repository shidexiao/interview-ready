import asyncio


async def slow_operation(timeout):
    print("Starting slow operation")
    try:
        await asyncio.sleep(2)
        print("Slow operation completed")
        return "Success"
    except asyncio.CancelledError:
        print("Slow operation cancelled")
        raise


async def main():
    try:
        # 设置1秒超时
        result = await asyncio.wait_for(slow_operation(2), timeout=1)
        print("Result:", result)
    except asyncio.TimeoutError:
        print("Operation timed out")

    # 另一种超时方式
    task = asyncio.create_task(slow_operation(2))
    try:
        result = await asyncio.shield(asyncio.wait_for(task, timeout=1))
        print("Shield result:", result)
    except asyncio.TimeoutError:
        print("Shield timed out, but task continues")
        await task  # 等待任务完成
        print("Task finally completed with:", task.result())


asyncio.run(main())