import asyncio

from corporatebench import CorporateBench, OpenAIBackend


async def main():
    async with CorporateBench(company="zenith") as bench:
        result = await bench.run(
            OpenAIBackend("gpt-5-nano"),
            method="rag",
            limit=10,
        )
        print(result.score)


if __name__ == "__main__":
    asyncio.run(main())
