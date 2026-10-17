import asyncio
from fastmcp import Client

async def main():
    async with Client(
        "https://suitable-chocolate-caterpillar.fastmcp.app/mcp",
        auth="oauth",
    ) as client:
        tools = await client.list_tools()

        for tool in tools:
            print(tool.name, "-", tool.description)

asyncio.run(main())