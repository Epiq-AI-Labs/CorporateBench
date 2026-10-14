import asyncio

from corporatebench import CorporateBench, OpenAIBackend


async def main():
    model = OpenAIBackend("gpt-5-nano", max_output_tokens=500)
    async with CorporateBench(company="biocure") as bench:
        result = await bench.run_topic_classification(
            model,
            shots=(0,),
            repetitions=1,
            test_size=31,
            max_requests=31,
        )
        print(result.scores_by_shots)


if __name__ == "__main__":
    asyncio.run(main())
