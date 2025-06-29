import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from dotenv import load_dotenv
from agents.run import RunConfig
import chainlit as cl

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
     model="gemini-2.0-flash",
    openai_client= external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="assistant",
    instructions="you are help ful assistant",
    model=model
)

@cl.on_chat_start
async def on_chat():
    await cl.Message("hello ask me something").send()

@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message("thinknkg...").send()

    result = Runner.run_sync(agent, message.content, run_config=config)
    await cl.message(result.final_output).send()        

 





