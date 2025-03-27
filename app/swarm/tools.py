from langchain_aws import BedrockEmbeddings
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain.prompts import PromptTemplate
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from typing import Annotated
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from .utils import (
    retriever,
    extract_n_load_relevant_info,
    CHUNK_SEPERATOR
)

embedding_model = BedrockEmbeddings(
    region_name="us-east-1",
    model_id="cohere.embed-english-v3",
    model_kwargs = {"input_type": "search_query"}
)

# Tools
@tool
def search_tool(query: str) -> str:
    """ Searches the vectorstore for relevant documents
    Args:
        query: The query to search for
    """
    docs = retriever.get_relevant_documents(query)
    return CHUNK_SEPERATOR.join([doc.page_content for doc in docs])

@tool
def web_search_tool(query: str) -> str:
    """ Searches the web for real time information and return relevant results to MongoDB Atlas so that the RAG model can access it through 'search_tool'
    Args:
        query: The query to search for
    """
    extract_n_load_relevant_info(query)
    return search_tool(query)

def create_grader_handoff_tool(*,model: BaseChatModel , generate_agent_name: str, rewrite_agent_name: str, tool_name: str, tool_description: str) -> BaseTool:

    def grade(question, docs):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            question (str): The user question
            docs (str): The retrieved documents

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECK RELEVANCE---")

        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM with tool and validation
        llm_with_tool = model.with_structured_output(grade)

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        # Chain
        chain = prompt | llm_with_tool
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score
        return score

    @tool(description=tool_description)
    def handoff_to_agent(
        # you can add additional tool call arguments for the LLM to populate
        # for example, you can ask the LLM to populate a task description for the next agent
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context."],
        # you can inject the state of the agent that is calling the tool
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        
        last_agent_message = state["messages"][-1]
        retrieved_messages = state["messages"][-2]
        question = last_agent_message.content
        docs = retrieved_messages.content
        positive_tool_message = ToolMessage(
            content=f"Successfully transferred to {generate_agent_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )

        negative_tool_message = ToolMessage(
            content=f"Successfully transferred to {rewrite_agent_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )

        positive_command = Command(
            goto=generate_agent_name,
            graph=Command.PARENT,
            # NOTE: this is a state update that will be applied to the swarm multi-agent graph (i.e., the PARENT graph)
            update={
                "messages": [last_agent_message, positive_tool_message],
                "active_agent": generate_agent_name,
                # optionally pass the task description to the next agent
                "task_description": task_description,
            },
        )

        negative_command = Command(
            goto=rewrite_agent_name,
            graph=Command.PARENT,
            # NOTE: this is a state update that will be applied to the swarm multi-agent graph (i.e., the PARENT graph)
            update={
                "messages": [last_agent_message, negative_tool_message],
                "active_agent": rewrite_agent_name,
                # optionally pass the task description to the next agent
                "task_description": task_description,
            },
        )
        
        if grade(question, docs) == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return positive_command
        elif grade(question, docs) == "no":
            print("---DECISION: DOCS NOT RELEVANT <REWRITE>---")
            return negative_command
        else:
            print("---DECISION: DOCS NOT RELEVANT <AGENT>---")
            return negative_command

    return handoff_to_agent