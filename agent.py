from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
import requests
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key="",
    base_url="https://api.groq.com/openai/v1",  # Any OpenAI-compatible endpoint
)
@tool
def get_url_content(url: str) -> str:
    """
    Fetch the raw content from a given URL using an HTTP GET request.

    Parameters
    ----------
    url : str
        The URL to fetch.

    Returns
    -------
    str
        The text content of the URL's response.

    Notes
    -----
    - Follows redirects automatically.
    - Returns plain text if available; binary data will be returned as bytes decoded with 'utf-8' (errors ignored).
    - Any exceptions are caught and returned as error messages.
    """
    try:
        response = requests.get(url,timeout=5)
        response.raise_for_status()
        
        # Try text decoding
        content_type = response.headers.get("Content-Type", "").lower()
        if "text" in content_type or "json" in content_type:
            return response.text
        else:
            # For binary files, return a short message instead of raw bytes
            return f"[Non-text content: {content_type}, {len(response.content)} bytes]"
    
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

class State(TypedDict):
    messages: Annotated[list,add_messages]

def create_graph():
    tools = [get_url_content]
    memory = MemorySaver()
    llm_with_tools = llm.bind_tools(tools)
    async def llm_node(state: State):
        messages = state["messages"]

        # Add system prompt only if not already present
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_prompt = SystemMessage(content="""
            Respond ONLY in this JSON format: {\"answers\": [answer1, answer2, ...]}
            Do what the user says, answer on the basis of the provided context, summarize the answer in 1 single line when using tools dont directly return it,THE ANSWER SHOULD BE IN THE SAME LANGUAGE AS THE QUESTION,
            if the context has factually incorrect information based on your knowledge then state that to the user, Dont do this for latest information from 2025 and beyond""")
            messages.insert(0, system_prompt)
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    builder = StateGraph(State)
    builder.add_node("llm_with_tools", llm_node)
    tool_node = ToolNode(tools=tools,handle_tool_errors=True)
    builder.add_node("tools", tool_node)
    builder.add_conditional_edges("llm_with_tools", tools_condition)
    builder.add_edge("tools", "llm_with_tools")
    builder.add_edge(START, "llm_with_tools")
    builder.add_edge("llm_with_tools", END)
    return builder.compile(checkpointer=memory)

