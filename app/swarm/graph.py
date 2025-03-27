from .tools import search_tool, web_search_tool, create_grader_handoff_tool
from .utils import (
    memory_saver)
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from functools import lru_cache
from langchain_core.language_models import BaseChatModel


george_prompt = """ 
You are an assistant by the Name George for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {state["messages"][-1].content} 
Context: {state["messages"][-2].content} 
Answer:
"""


def initialize_swarm_graph(model: BaseChatModel):
    eve = create_react_agent(
        model,
        [create_handoff_tool(agent_name="Bob", description="Eve can ask Bob to find any kind of information"), 
         create_grader_handoff_tool(model=model, generate_agent_name="George", rewrite_agent_name="Bob", tool_name="Grader tool", tool_description="Eve can ask to George to generate the final answer to users question if the retrieved documents are relevant to the question")
        ],
        prompt="You are Eve, an expert planning agent who can plan, rewrite questions and execute tasks. Do not make up answers and always rely on tools to fetch relevant information.",
        name="Eve",
    )

    bob = create_react_agent(
        model,
        [web_search_tool,
        create_handoff_tool(agent_name="Eve", description="Bob can ask Eve to plan or rewrite queries to extract more relavant information and rephrase the response from the retriever to get more relevant answers"),
        create_grader_handoff_tool(model=model, generate_agent_name="George", rewrite_agent_name="Eve", tool_name="Grader tool", tool_description="Bob can ask to George to generate the final answer to users question if the the retrieved documents are relevant to the question")
        ],
        prompt="You are Bob, you have access to web search and find information that you cannot find in private knowledgebase",
        name="Bob",
    )

    george = create_react_agent(
        model,
        [],
        prompt=george_prompt,
        name="George",
    )
    # Create the workflow
    workflow = create_swarm(
        [eve, bob, george],
        default_active_agent="Eve"
    )
    app = workflow.compile(checkpointer=memory_saver)
    return app
