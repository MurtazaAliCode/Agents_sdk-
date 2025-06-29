import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from dotenv import load_dotenv
from agents.run import RunConfig

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
     model="gemini-2.0-flash",
    openai_client= external_client,
   frequency_penalty=0
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
    
)

agent = Agent(
    name="assistant",
    instructions="you are help ful assistant",
    model=model
   
)

result = Runner.run_sync(agent,"who is the founder of PIAIC ",run_config=config)
print(result.final_output)