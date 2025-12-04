from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatOpenAI(model="gpt-4o-mini")

# Crear el agente con LangGraph
agent = create_react_agent(
    model=model,
    tools=[get_weather],
)