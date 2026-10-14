import asyncio

from corporatebench import CorporateBench, OpenAIBackend


async def main():
    async with CorporateBench(company="zenith") as bench:
        result = await bench.run_kb_evaluation(
            OpenAIBackend("gpt-5-nano"),
            max_requests=bench.kb_evaluation_requests,
        )
        print(result.entities.macro_f1)


if __name__ == "__main__":
    asyncio.run(main())
