# 基础协程示例
import asyncio

async def hello_world():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

async def main():
    await asyncio.gather(hello_world(), hello_world())

asyncio.run(main())
