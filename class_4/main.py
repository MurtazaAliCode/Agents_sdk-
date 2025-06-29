from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

@function_tool
async def get_weather(city: str) -> str:
    print("get weather called")
    return f"The weather in {city} is sunny"

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You are a translator agent. Your job is to translate the provided text into Spanish.",
    model=model,
)

french_agent = Agent(
    name="French agent",
    instructions="You are a translator agent. Your job is to translate the provided text into French.",
    model=model,
)

chinese_agent = Agent(
    name="Chinese Agent",
    instructions="You are a translator agent. Your job is to translate the provided text into Chinese.",
    model=model,
)

urdu_agent = Agent(
    name="Urdu Agent",
    instructions="You are a translator agent. Your job is to translate the provided text into urdu.",
    model=model,
)

async def main():
    agent = Agent(
        name="main agent",
        instructions="You are a triage agent. Follow these rules: "
                    "1. If the user asks to translate to Spanish, use spanish_agent. "
                    "2. If the user asks to translate to French, use french_agent. "
                    "3. If the user asks to translate to urdu, use urdu_agent. "
                    "4. If the user asks to translate to Chinese, use chinese_agent.",
        tools=[
            spanish_agent.as_tool(
                tool_name="translate_to_spanish",
                tool_description="Translates the provided text into Spanish",
            ),
            french_agent.as_tool(
                tool_name="translate_to_french",
                tool_description="Translates the provided text into French",
            ),
            chinese_agent.as_tool(
                tool_name="translate_to_chinese",
                tool_description="Translates the provided text into Chinese",
            ),
             urdu_agent.as_tool(
                tool_name="translate_to_urdu",
                tool_description="Translates the provided text into urdu",
            )
        ],
        model=model,  
    )

    result = await Runner.run(agent, 'translate this to urdu: hello how are you', run_config=config)

    print("Final Response:", result.final_output)
    print("Last Agent:", result.last_agent.name)

if __name__ == "__main__":
    asyncio.run(main())