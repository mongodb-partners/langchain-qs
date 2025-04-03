from langchain_aws import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from typing import Annotated
from .utils import CHUNK_SEPERATOR, embedding_model, retriever
from langchain_core.language_models import BaseChatModel

from pydantic import BaseModel, Field

# Tools
@tool
def search_tool(query: str) -> str:
    """ Searches the vectorstore for relevant documents
    Args:
        query: The query to search for
    """
    docs = retriever.get_relevant_documents(query)
    return CHUNK_SEPERATOR.join([doc.page_content for doc in docs])

def create_rewrite_handoff_tool(*, model: BaseChatModel,  agent_name: str, tool_name: str, tool_description: str) -> BaseTool:
    
    def query_rewrite_tool(query:str) -> str:
        """ Rewrites the query to be more specific
        Args:
            query: The query to rewrite
        """
        # LLM with tool and validation
        llm_with_tool = model

        # Prompt
        prompt = PromptTemplate(
            template="""You are a query rewriter. \n 
            Here is the user question: {query} \n
            Rewrite the question to be more specific.""",
            input_variables=["query"],
        )

        # Chain
        chain = prompt | llm_with_tool | StrOutputParser()
        rewritten_query = chain.invoke({"query": query})
        return rewritten_query
    
    @tool(description=tool_description)
    def handoff_to_agent(
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context."],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        print("---REWRITE QUERY HANDOFF TO AGENT---")
        last_agent_message = state["messages"][-1]
        question = last_agent_message.content
        rewritten_query = query_rewrite_tool(question)
        tool_message = ToolMessage(
            content=f"Query: {rewritten_query} \nAction:Successfully transferred to {agent_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        command = Command(
            goto=agent_name,
            graph=Command.PARENT,
            # NOTE: this is a state update that will be applied to the swarm multi-agent graph (i.e., the PARENT graph)
            update={
                "messages": [last_agent_message, tool_message],
                "active_agent": agent_name,
                # optionally pass the task description to the next agent
                "task_description": task_description,
            },
        )
        return command
    return handoff_to_agent

def create_final_response_handoff_tool(*, model: BaseChatModel, agent_name: str, tool_name: str, tool_description: str) -> BaseTool:
    
    def generate_final_response_tool(question: str, docs: str) -> str:
        """ Generates the final response to the user
        Args:
            task_description: The task description to generate the final response
        """
        # LLM with tool and validation
        llm_with_tool = model

        # Prompt
        prompt = PromptTemplate(
            template="""You are a final response generator. \n 
            Here is the user question: {question} \n
            Here are the retrieved documents: {docs} \n
            Generate the final response to the user.""",
            input_variables=["question", "docs"],
        )

        # Chain
        chain = prompt | llm_with_tool | StrOutputParser()
        answer_response = chain.invoke({"question": question, "docs": docs})
        return answer_response
    
    @tool(description=tool_description)
    def handoff_to_agent(
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context."],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        print("---FINAL RESPONSE HANDOFF TO AGENT---")
        last_agent_message = state["messages"][-1]
        final_response = generate_final_response_tool(
            last_agent_message.content,
            state["messages"][-2].content,
        )
        tool_message = ToolMessage(
            content=f"Final Response: {final_response} \nAction: Successfully transferred to {agent_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        command = Command(
            goto=agent_name,
            graph=Command.PARENT,
            # NOTE: this is a state update that will be applied to the swarm multi-agent graph (i.e., the PARENT graph)
            update={
                "messages": [last_agent_message, tool_message],
                "active_agent": agent_name,
                # optionally pass the task description to the next agent
                "task_description": task_description,
            },
        )
        return command
    return handoff_to_agent

@tool
def web_search_tool(query: str) -> str:
    """ Searches the web for real time information/ information you do not have and return relevant results
    Args:
        query: The query to search for
    """
    extract_n_load_relevant_info(query)
    return search_tool(query)

# from typing import Annotated


def create_grader_handoff_tool(*,model: BaseChatModel, generate_agent_name: str, rewrite_agent_name: str, tool_name: str, tool_description: str) -> BaseTool:

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
        print("I am here")
        scored_result = chain.invoke({"question": question, "context": docs})
        print(f"Graded result: {scored_result}")
        score = scored_result.binary_score
        print(f"Graded result: {score}")
        return score

    @tool(description=tool_description)
    def handoff_to_agent(
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context."],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        print("---Grader HANDOFF TO AGENT---")
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


