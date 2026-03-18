"""测试回调网络连接"""
import asyncio
import httpx

async def test_connection():
    urls = [
        "http://localhost:8092/api/v1/callbacks/liveness/action",
        "http://host.docker.internal:8092/api/v1/callbacks/liveness/action",
        "http://172.17.0.1:8092/api/v1/callbacks/liveness/action",
    ]
    
    for url in urls:
        try:
            print(f"\n测试连接: {url}")
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url.replace("/callbacks/liveness/action", "/healthz"))
                print(f"  ✓ 连接成功: {response.status_code}")
        except Exception as e:
            print(f"  ✗ 连接失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
