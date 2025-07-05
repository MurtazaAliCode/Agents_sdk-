import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, Tool, Handoff, ModelSettings
from dotenv import load_dotenv
from agents.run import RunConfig
from typing import List, Callable, Any

# Load environment variables
load_dotenv()

# Retrieve API key
api_key = os.getenv("GEMINI_API_KEY")

# Initialize AsyncOpenAI client
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Configure the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
    # frequency_penalty=0,
    temperature=0.7  # Added for better response variety
)

# Define a simple calculator tool
def calculate_tool(input_data: dict) -> str:
    operation = input_data.get("operation")
    num1 = float(input_data.get("num1", 0))
    num2 = float(input_data.get("num2", 0))
    if operation == "add":
        return str(num1 + num2)
    elif operation == "square":
        return str(num1 ** 2)
    return "Invalid operation"

calculator_tool = Tool(
    name="calculator",
    description="Performs basic calculations like addition and square.",
    func=calculate_tool,
    input_schema={"type": "object", "properties": {"operation": {"type": "string"}, "num1": {"type": "number"}, "num2": {"type": "number"}}}
)

# Define a handoff to an expert agent
def expert_handoff(context: Any, agent: Any) -> str:
    return "Transferring to AI expert for detailed analysis."

expert_handoff_instance = Handoff(
    name="expert_handoff",
    description="Transfers to an expert agent for complex queries.",
    func=expert_handoff
)

# Set up the run configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
    timeout=30  # Added timeout for better control
)

# Create an agent with enhanced parameters
agent = Agent(
    name="AI_Expert",
    instructions="Aap AI ke expert hain aur detail mein jawab dete hain. Agar sawal complex ho to expert ko haath den.",
    prompt="Detail mein jawab den, agar zaroori ho to calculator use karen.",
    handoff_description="Complex AI queries ko expert agent ko haath dena.",
    handoffs=[expert_handoff_instance],
    model="gemini-2.0-flash",
    model_settings=ModelSettings(temperature=0.7, max_tokens=500),  # Enhanced settings
    tools=[calculator_tool],
    mcp_servers=[],  # Placeholder, add if required
    # mcp_config=lambda: MCPConfig(),  # Default config
    input_guardrails=[]  # Placeholder, add if required
)

# Run a query synchronously
result = Runner.run_sync(agent, "who is the founder of PIAIC? OR calculate square of 5", run_config=config)
print(result.final_output)