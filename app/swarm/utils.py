from dotenv import load_dotenv

load_dotenv()
import os

from pymongo import MongoClient
import certifi
from pymongo.operations import SearchIndexModel
from time import sleep
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models import BaseChatModel
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_fireworks import ChatFireworks
from langgraph.checkpoint.mongodb import MongoDBSaver

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage,BaseMessage, ToolMessage, AIMessage

from functools import lru_cache

embedding_model = BedrockEmbeddings(
    region_name="us-east-1",
    model_id="cohere.embed-english-v3",
    model_kwargs = {"input_type": "search_query"}
)

def get_llm(model_name: str = "accounts/fireworks/models/llama-v3p1-405b-instruct") -> BaseChatModel:
    if "fireworks" in model_name:
        llm = ChatFireworks(model=model_name)
    else:
        llm = ChatBedrock(model=model_name)
    return llm

EMBED_DIMENSION = len(embedding_model.embed_documents(["hello world"])[0])
print(f"Embedding dimension: {EMBED_DIMENSION}")


# retriever
CHUNK_SEPERATOR = "\n\n~~~~~~~~~~~~\n\n"
client = MongoClient(
    os.getenv("MONGODB_URI"),
    tlsCAFile=certifi.where()
)
db = client[os.getenv("MONGODB_DB")]
collection = db[os.getenv("MONGODB_COLLECTION")]

# Create search index
search_index_model = SearchIndexModel(
        name="default",
        type="vectorSearch",
        definition={
            "fields": [
                {
                    "path": "embedding",
                    "type": "vector",
                    "numDimensions": EMBED_DIMENSION,
                    "similarity": "cosine"
                }
            ]
        }
    )

def get_index_config(index_name: str):
    idxs = list(collection.list_search_indexes())
    for ele in idxs:
        if ele["name"]==index_name:
            return ele
    return None

def create_search_index(index_name: str):
    #check if index exists
    idx = get_index_config(index_name)
    if idx and idx["queryable"]:
        print("Vector search index already exists.")
        return
    collection.create_search_index(search_index_model)
    while True:        
        idx = get_index_config(index_name)
        print(idx)
        if idx["queryable"]:
            print("Vector search index created successfully.")
            break
        print("Please wait!!! Creating vector search index...")
        sleep(5)

def get_all_exisiting_sources():
    sources = collection.aggregate([
        {
            '$group': {
                '_id': '$source', 
                'source': {
                    '$first': '$source'
                }
            }
        }, {
            '$project': {
                'source': 1, 
                '_id': 0
            }
        }
    ])
    return set([ele["source"] for ele in sources])

# vectorstore

vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    text_key="content",
    index_name="default",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ckpt_db_name = os.getenv("MONGODB_CKPT_DB_NAME", "test_ckp_2")
ckpt_collection_name = os.getenv("MONGODB_CKPT_COLL_NAME","test_ckpt_2")
# Memory checkpoint saver for graph
memory_saver = MongoDBSaver(client, ckpt_db_name, ckpt_collection_name)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

### Utils methods
def extract_n_load_relevant_info(query: str):
    """ Searches for relevant realtime information and then extracts text from URL sources and loads it into the vectorstore 
        Works as memory for the LLM Agent
    Args:
        url: The URL to extract text from
    """
    res = TavilySearchResults(max_results=5).run(query)
    urls = [ele["url"] for ele in res]
    urls = list(set(urls) - get_all_exisiting_sources())
    print(f"Total urls to process: {len(urls)}")
    docs = []
    for url in urls:
        loader = WebBaseLoader(web_path=url)
        # Load the data from the website
        docs += loader.load()
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
    # create search index if it does not exist
    create_search_index("default")
    print(f"Total chunks: {len(chunks)}")

def load_vectorstore_from_url(url: str):
    loader = WebBaseLoader(web_path=url)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
    # create search index if it does not exist
    create_search_index("default")
    print(f"Total chunks: {len(chunks)}")