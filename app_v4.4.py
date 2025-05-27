# MAY-23-2025-FRIDAY==========================================================================================
from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing_extensions import TypedDict, List, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from utils import search_and_parse, just_parse, calculator_add, calculator, max_min,cleanup_doc,empty_cache
import time
from typing import Annotated
from langchain_core.prompts import PromptTemplate
import os
import pymupdf
# Last Updated: MAY-27, 5:49pm

# Initialization --------------------------------------------------------------------------------------------
pdf_fp = r"C:\Users\guacw\Downloads\EPA-HQ-OPPT-2019-0263-0038_attachment_11.pdf" # put pdf here
vectordb_dir = r'.\vector_db'
cache_dir = r'.\cache'

# Define the local QNA model -------------------------------------------------------------------------------------
llm = LlamaCpp(
    model_path = r".\models\Llama-3.1-8B-Instruct\Meta-Llama-3.1-8B-Instruct-f32.gguf",
    n_ctx = 131072,
    n_threads = 8,
    n_batch = 128,
    temperature = 0.5,
    verbose = False,
    #n_gpu_layers = 100 # layers to route to gpu
)
# Define the local Tool Calling model ---------------------------------------------------------------------------
llm_tool = LlamaCpp(
    model_path = r".\models\phi-4-reasoning-plus\Phi-4-reasoning-plus-Q2_K.gguf",
    n_ctx = 32768,
    n_threads = 8,
    n_batch = 128,
    temperature = 0.5,
    verbose = False,
    #n_gpu_layers = 100
)

# Initialize State Schemas --------------------------------------------------------------------------------------
class GraphState(TypedDict):
    question:str
    augmented_question:str
    keywords:List[str]
    context:List[Document]
    tool_result:str
    output:str 
    messages: Annotated[list[AnyMessage], add_messages] 
    vector_store: any 
    tool_message: str
    #vector_stores: dict # for multiple pdfs. { pdf_fp: vector_store }

# Define Nodes --------------------------------------------------------------------------------------------------------
def math_tools_call(state:GraphState):

    template1 = f"""
    \nYou are a tool selector. Your job is to analyze the user's question and decide if it requires a math operation.
    \n
    \nIf it does, output 'calculation required' No extra text, no reasoning. Only this line. 
    \nIf it does not, output 'no tools needed'. No extra text, no reasoing. Only this line. 

    \nExample acceptable outputs:
    \n no tools needed 
    \n calculation required 
    \n
    \nExample unacceptable outputs:
    \nHere is your answer: no tools needed
    \nHere is the tool call result: no tools needed
    \nTherefore: calculated required
    \n"no tools needed"
    \n
    \nThe question:
    {question}
    """
    call_init = llm_tool.invoke(template1).strip()
    call_result = None

    if call_init == 'calculation required':
        template2 = f"""
        In a previous step, it was determined that this question may only be answered with a calculation.
        \n 
        \nYour job is to read the question to determine the type of calculation needed. And then read the context to extract the exact numbers required for the calculation.
        \nAnswer only in this exact format: '<operation>, <numbers>' where <operation> can only adoopt one of the following EXACT strings: 
        \n'calculate_add','calculate','max_min'
        \nand <numbers> must be replaced with actual numbers for the calculation, separated by commas. 
        \nHere are example outupts: 
        \n calculate,12,24,32,13,45,32
        \n max_min,12,33,24,54,32
        \n calculate_add,1,54
        \n 
        \n{state['augmented_question']}
        """
        call_result = llm_tool.invoke(template2).strip()
    else:
        call_result = 'no tools needed'

    return {'tool_message':call_result}

def math_tools_use(state:GraphState):
    tool_call = state['tool_message']
    result = None 

    if tool_call == 'no tools needed':
        return {'tool_result':None}
    else:
        try:
            tool_parts = tool_call.split(',')
            tool_name = tool_parts[0]
            if tool_name == 'calculator_add':
                nums = list(map(float, tool_parts[1:]))
                result = calculator_add(*nums)
            elif tool_name == "calculator":
                operation = tool_parts[1]
                nums = list(map(float, tool_parts[2:]))
                result = calculator(operation, *nums)
            elif tool_name == "max_min":
                operation = tool_parts[1]
                nums = list(map(float, tool_parts[2:]))
                result = max_min(operation, *nums)
        except Exception as e:
            raise Exception(f"Tool call failed: {e}\nCommand: {tool_call}")
    
    return {'tool_result':str(result)}
    
def retrieve(state: GraphState):
    if state['vector_store'] is None:
        raise ValueError("Vector store not initialized")
    retrieved_docs = state['vector_store'].similarity_search(state["question"])
    # similarity search is not so great; maybe try again with better embedding function in the future; or split input into further keywords?
    # for now, I'm going to keep going. (MAY-25)

    ''' 
    # For multiple pdfs
    vector_store = state['vector_stores'].get(pdf_fp)
    if vector_store is None:
        raise ValueError('Vector store is not initialized')
    retrieved_docs = vector_store.similarity_search(state['question'])
    '''

    return {'context': retrieved_docs}

def augment(state:GraphState):
    # Clean up 
    docs = cleanup_doc(state['context']) # [Document,Document...]
    texts = '\n\n'.join(doc.page_content for doc in docs)

    question = state['question']
    input_text = f"""
    Question:\n {question}\n 
    Context:\n {texts}
    """

    return{'augmented_question':input_text}

def generate(state:GraphState):
    question = state['augmented_question']
    tool_msg = state['tool_message']
    tool_result = state['tool_result']

    final_input = f"""
    You are an assistant for question-answering tasks. \n
    Use the following pieces of retrieved context to answer the question.\n 
    \n{question}
    \nTool Used (if Any): {tool_msg}
    \nTool Result (if Any): {tool_result}
    \n
    \nStop IMMEDIATELY after reaching the answer to the question. Make sure to follow all guidelines in the question.'
    """
    
    output = llm.invoke(final_input)
    conversation_history = [
        HumanMessage(final_input,name='Human'),
        AIMessage(output,name='D.A.T.A')
    ]

    return{
        'output':output,
        'messages':conversation_history
    }

# AI Router chooses Parsing function------------------------------------------------------------------------------------------------------
'''
llm_parse_tools = llm.bind_tools([search_and_parse,just_parse])
def parsing_tool_call(state:GraphState):
    return{"messages":[llm_parse_tools.invoke(state["question"])]}

tool_parse = ToolNode(tools=[search_and_parse,just_parse])
'''
# Manual choosing of Parsing function ----------------------------------------------------------------------------------------------------
def parse_snp(state:GraphState):
    vector_store = search_and_parse(pdf_fp,state['keywords'],vectordb_dir)
    return {'vector_store':vector_store}
    #return {'vector_stores': {pdf_fp:vector_store}} # for multiple pdfs

def parse_all(state:GraphState):
    vector_store = just_parse(pdf_fp,vectordb_dir)
    return{'vector_store':vector_store}
    #return {'vector_stores': {pdf_fp:vector_store}} # for multiple pdfs

def snp_condition(state:GraphState) -> Literal['parse_snp','parse_all']:
    if state['keywords'] and any(k is not None for k in state['keywords']):
        return 'search_and_parse'
    return 'just_parse'

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
builder.add_node('math_tools_call',math_tools_call)
builder.add_node('math_tools_use',math_tools_use)
builder.add_node('generate_1',generate)

#Edges
builder.add_conditional_edges(START, snp_condition, {
    'search_and_parse': 'search_and_parse',
    'just_parse': 'just_parse'
})
builder.add_edge('search_and_parse', 'retriever_1')
builder.add_edge('just_parse', 'retriever_1')
builder.add_edge('retriever_1', 'augment_1')
builder.add_edge('augment_1', 'math_tools_call')
builder.add_edge('math_tools_call','math_tools_use')
builder.add_edge('math_tools_use', 'generate_1')
builder.add_edge('generate_1', END)
graph = builder.compile()

#-----------------------------------------------Run Graph ------------------------------------------------
inputs = [
    ['question 1',['for the micronucleus assay, is it an in vivo study or an in vitro study? Answer ONLY in the format: Micronucleus - <in vitro/in vivo>. This question does not require any calculations.',['micronucleus']]],
    ['question 2',['for the micronucleus assay, what is the animal test system used in the study? Please answer with the animal type (e.g. mouse, rat,rabbit..etc.) and species (e.g. Sprague-Dawley) if possible. Answer ONLY in the format: <Animal> - <Species/Strain>. This question does not require any calculations.',['micronucleus','strain']]]
] # question:str, keywords:list()

for idx,q in enumerate(inputs):
    question_idx = q[0]
    question = q[1][0]
    keywords = q[1][1] 
    start_time = time.time()
    response = graph.invoke({
        'question':question,
        'keywords':keywords,
        'context':[],
        'tool_result':None,
        'output':None,
        'messages':[],
        'vector_store':None,
        'tool_message':None
    })
    with open(os.path.join(cache_dir,'run_2.txt'),'a',encoding='utf-8') as cache:
        records = [message.pretty_repr() for message in response['messages']]
        for r in records:
            cache.write(r + '\n')
    
    end_time = time.time()
    duration = end_time-start_time
    time_per_trial.append(duration)
    print(f'Trial {idx} complete, time took: {duration:.2f} seconds.')

    empty_cache(vectordb_dir)
# GraphState is a type, not instance.

llm.client.close()
