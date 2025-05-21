# In progress. 
# As of MAY-09-2025: Cleaned up initialization, removed messy and errornoeus sections. Developing tools for the AI. AI flow constructed. 

# This code is the core executable file for the MAMTOX AI 
# The model chosen for initial demonstration of the project is DeepSeek R1 1.5B, imported from huggingface 
# Ctrl + I for help

# MAY-21-2025, WED
# Initialization 
pdf_fp = r'' # put pdf here
question = "" # put question here
keywords = [] # put list of keywords (str) here

vectordb_dir = r'vectordb/'
cache_dir = r'cache/'

# Define the local model
llm = Llamacpp(
    model_path = r'models/DeepSeek-R1-Distill-Qwen-14B/DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf',
    n_ctx = 4096,
    n_threads = 8,
    n_batch = 128,
    temperature = 0,
    verbose = False
)

# Build the AI pipeline
import langgraph 
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
# from langgraph.graph import MessagesState 
from typing_extensions import TypedDict
from utils import search_and_parse, just_parse
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

llm_with_tools = llm.bind_tools([search_and_parse, just_parse]) # put tools here
def tools_call_1(state:MessagesState):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}


# Build Graph
builder = StateGraph(MessagesState)
builder.add_node("tools_call_1", tools_call_1) # node to call tool uses, no output answer
builder.add_edge(START,'tools_call_1')
builder.add_node('tools_use_1',ToolNode([search_and_parse, just_parse])) # put tools here, actually outputs answer
builder.add_conditional_edges("tools_call_1",tools_condition)
builder.add_edge("tools_use_1",END)

graph = builder.comile()

# Messages and inputs
messages_first = [HumanMessage(content = "", name='Human')] # put input here

messages_to_add = [HumanMessage(content = "Keywords for you to parse the pdf into relevant sections to answer the input question: ")]
add_messages(messages_first,messages_to_add)

messages = graph.invoke({"messages":messages}) # Gathers first AI response. 

messages['messages'][-1].pretty_print() # Prints first Ai resopnse? 
