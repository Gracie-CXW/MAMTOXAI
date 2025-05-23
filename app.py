# MAY-23-2025-FRIDAY==========================================================================================
from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing_extensions import TypedDict, List, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from utils import search_and_parse, just_parse, calculator_add, calculator_multiply
import time

# Initialization --------------------------------------------------------------------------------------------
pdf_fp = r'' # put pdf here
vectordb_dir = r'vectordb/'
cache_dir = r'cache/'

# Define the local model -------------------------------------------------------------------------------------
llm = Llamacpp(
    model_path = r'models/DeepSeek-R1-Distill-Qwen-14B/DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf',
    n_ctx = 4096,
    n_threads = 8,
    n_batch = 128,
    temperature = 0,
    verbose = False,
    n_gpu_layers = 40 # layers to route to gpu
)

# Initialize State Schemas ---------------------------------------------------
class GraphState(TypedDict):
    question:str
    keywords:List[str]
    context:List[Document]
    tool_result:str
    output:str 
    messages: Annotated[list[AnyMessage], add_messages]
    vector_store: Any 
    #vector_stores: dict # for multiple pdfs. { pdf_fp: vector_store }

# Define Nodes ----------------------------------------------------------------
llm_with_tools = llm.bind_tools([calculator_add,calculator_multiply]) # put tools here
def tools_call_1(state:GraphState):
    return {"messages":[llm_with_tools.invoke(state["question"])]}

tool_node_1 = ToolNode(tools=[calculator_add,calculator_multiply]) # tools here

def retrieve(state: GraphState):
    if state['vector_store'] is None:
        raise ValueError("Vector store not initialized")
    retrieved_docs = state['vector_store'].similarity_search(state["question"])

    ''' 
    # For multiple pdfs
    vector_store = state['vector_stores'].get(pdf_fp)
    if vector_store is None:
        raise ValueError('Vector store is not initialized')
    retrieved_docs = vector_store.similarity_search(state['question'])
    '''

    return {'context': retrieved_docs}

def augment(state:GraphState):
    texts = '\n\n'.join(doc.page_content for doc in GraphState['context'])
    question = state['question']
    input_text = f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Question: {question} \n Context: {texts}"

    return{'question':input_text}

def generate(state:GraphState):
    output = llm.invoke(state['question'])
    conversation_history = [
        HumanMessage(state['question'],name='Human'),
        AIMessage(output,name='D.A.T.A')
    ]
    return{
        'output':output,
        'messages':conversation_history
    }

# AI Router chooses Parsing function----------------------------
'''
llm_parse_tools = llm.bind_tools([search_and_parse,just_parse])
def parsing_tool_call(state:GraphState):
    return{"messages":[llm_parse_tools.invoke(state["question"])]}

tool_parse = ToolNode(tools=[search_and_parse,just_parse])
'''
# Manual choosing of Parsing function -------------------------
def parse_snp(state:GraphState):
    search_and_parse(pdf_fp,state['keywords'],vectordb_dir)
    return {'vector_store':vector_store}
    #return {'vector_stores': {pdf_fp:vector_store}} # for multiple pdfs

def parse_all(state:GraphState):
    just_parse(pdf_fp,vectordb_dir)
    return{'vector_store':vector_store}
    #return {'vector_stores': {pdf_fp:vector_store}} # for multiple pdfs


def snp_condition(state:GraphState) -> Literal['parse_snp','parse_all']:
    if state['keywords'] and any(k is not None for k in state['keywords']):
        return 'parse_snp'
    return 'parse_all'

# Set up counter ----------------------------------------------------------------------------------------
time_per_trial = []
time_per_run = [] # for multiple runs (multiple pdfs)
#-----------------------------------------------------Build Graph ---------------------------------------
builder = StateGraph(GraphState)
# Nodes
builder.add_node('search_and_parse',parse_snp)
builder.add_node('just_parse',parse_all)
builder.add_node("retriever_1",retrieve)
builder.add_node('augment_1',augment)
builder.add_node('tool_call_1',tools_call_1)
builder.add_node('tool_use_1',tool_node_1)
builder.add_node('generate_1',generate)

#Edges
builder.add_conditional_edges(START, snp_condition, {
    'search_and_parse': 'search_and_parse',
    'just_parse': 'just_parse'
})
builder.add_edge('search_and_parse', 'retriever_1')
builder.add_edge('just_parse', 'retriever_1')
builder.add_edge('retriever_1', 'augment_1')
builder.add_edge('augment_1', 'tool_call_1')
builder.add_conditional_edges('tool_call_1', tools_condition)
builder.add_edge('tool_use_1', 'generate_1')
builder.add_edge('generate_1', END)
graph = builder.compile()

#-----------------------------------------------Run Graph ------------------------------------------------
inputs = [
    ['question 1',[]],
    ['question 2',[]]
] # question:str, keywords:list()

for idx,q in enumerate(inputs):
    question = q[0]
    keywords = q[1] 
    start_time = time.time()
    response = graph.invoke({'question':question,'keywords':keywords})
    with open(os.path.join(cache_dir,'run_1.txt'),'a') as cache:
        cache.write(response['messages'] + '\n')
    
    end_time = time.time()
    duration = end_time-start_time
    time_per_trial.append(duration)
    print('Trial {idx} complete, time took: {duration:.2f} seconds.')
# GraphState is a type, not instance.

