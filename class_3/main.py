import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
import chainlit as cl

# Custom agents package (assumed from your structure)
from agents import Agent, Runner, OpenAIChatCompletionsModel, InputGuardrail, GuardrailFunctionOutput, AsyncOpenAI
from agents.run import RunConfig

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Define external client (not used directly in this example)
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Output type for guardrail agent
class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

# ✅ Agent definitions — Corrected indentation and placement
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check karein ke user homework ke baray mein pooch raha hai ya nahi.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Math ke sawalaat ke liye mahir agent",
    instructions="Aap math ke masail mein madad karte hain. Har qadam par apni wajahat bayan karein aur misalein shamil karein.",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Tareekhi sawalaat ke liye mahir agent",
    instructions="Aap tareekhi sawalaat mein madad faraham karte hain. Aham waqiat aur background ko wazeh taur par bayan karein.",
)

# ✅ Guardrail function
async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

# ✅ Triage agent that routes to subject agents
triage_agent = Agent(
    name="Triage Agent",
    instructions="Aap user ke homework ke sawal ki bunyad par yeh tay karte hain ke kaun sa agent istemal karna hai.",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[InputGuardrail(guardrail_function=homework_guardrail)]
)

# ✅ Main async function
async def main():
    result = await Runner.run(triage_agent, "America ke pehle saddar kaun the?")
    print(result.final_output)

    # Optional second test
    # result = await Runner.run(triage_agent, "Zindagi kya hai?")
    # print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
