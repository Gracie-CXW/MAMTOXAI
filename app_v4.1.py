# MAY-23-2025-FRIDAY==========================================================================================
from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing_extensions import TypedDict, List, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from utils import search_and_parse, just_parse, calculator_add, calculator, max_min,cleanup_doc
import time
from typing import Annotated
from langchain_core.prompts import PromptTemplate
import os
import pymupdf
# Last Updated: MAY-27, 3:33pm

# Initialization --------------------------------------------------------------------------------------------
pdf_fp = r"C:\Users\guacw\Downloads\EPA-HQ-OPPT-2019-0263-0038_attachment_11.pdf" # put pdf here
vectordb_dir = r'.\vector_db'
cache_dir = r'.\cache'

# Define the local model -------------------------------------------------------------------------------------
llm = LlamaCpp(
    model_path = r".\models\Llama-3.1-8B-Instruct\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    n_ctx = 131072,
    n_threads = 8,
    n_batch = 128,
    temperature = 1.0,
    verbose = False,
    #n_gpu_layers = 40 # layers to route to gpu
)

# Initialize State Schemas ---------------------------------------------------
class GraphState(TypedDict):
    question:str
    keywords:List[str]
    context:List[Document]
    tool_result:str
    output:str 
    messages: Annotated[list[AnyMessage], add_messages] 
    vector_store: any 
    tool_message: str
    #vector_stores: dict # for multiple pdfs. { pdf_fp: vector_store }

# Define Nodes ----------------------------------------------------------------
def math_tools_call(state:GraphState):

    template = f"""
    [SYSTEM INSTRUCTIONS]
    You are a tool selector for mathematical operations. Your ONLY task is to:
    1. Analyze if the question requires a mathematical operation
    2. If yes, select the EXACT tool format needed
    3. If no, respond with "no tools needed"
    
    [RULES]
    - ONLY respond with one of these EXACT formats (nothing else):
      - calculator_add,<num1>,<num2>
      - calculator,<operation>,<num1>,<num2>
      - max_min,<operation>,<num1>,<num2>,<num3>
      - no tools needed

    - NEVER explain your reasoning
    - NEVER add extra text
    - If unsure, default to "no tools needed"
    
    [QUESTION ANALYSIS]
    Question: {state['question']}
    Context: {state['context']}
    
    [TOOL SELECTION OUTPUT]
    """

    call = llm.invoke(template).strip()

    return {'tool_message':call}

'''
def math_tools_use(state:GraphState): 
    tool_call = state['tool_message'][-1] # [str,str,str]
    result = None 
    #results = [] # For multiple entries 
    if state['tool_message'] is not None:
        for idx, entry in enumerate(tool_call):
            entry = entry.split(',') # Splits tool call entry (str) into a list. 
            if entry[0] == 'calculator_add':
                try:
                    nums = entry[1].split(',')
                    nums = [int(n) for n in nums]

                    result = entry[0].globals(*nums) # not sure if this syntax is okay...
                    #results.append([idx,entry[0],result])
                except Exception as e:
                    raise Exception(f"tool call failed: str{e}\n Call command: {tool_call}")
    
            elif entry[0] == 'calculator':
                try: 
                    operation = entry[1]
                    nums = entry[2].split(',')

                    result = calculator(operation,*nums)
                    #results.append([idx,entry[0],result])
                except Exception as e:
                    raise Exception(f"tool call failed: str{e} \n Call command: {tool_call}")

            elif entry[0] == 'max_min':
                try: 
                    operation = entry[1]
                    nums = entry[2].split(',')

                    result = max_min(*nums) 
                    #results.append([idx,entry[0],result])
                except Exception as e: 
                    raise Exception(f"tool call failed: str{e} \n Call command: {tool_call}")
            else:
                pass
        if result:
            return {'tool_result':str(result)}
        else:
            return {'tool_result': None}
    else:
        return {'tool_result': None}
'''
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
                nums = list(map(float, parts[1:]))
                result = calculator_add(*nums)
            elif tool_name == "calculator":
                operation = parts[1]
                nums = list(map(float, parts[2:]))
                result = calculator(operation, *nums)
            elif tool_name == "max_min":
                operation = parts[1]
                nums = list(map(float, parts[2:]))
                result = max_min(operation, *nums)
        except Exception as e:
            raise Exception(f"Tool call failed: {e}\nCommand: {tool_call}")
    
    return {'tool_result':str(result)}
    
def retrieve(state: GraphState):
    if state['vector_store'] is None:
        raise ValueError("Vector store not initialized")
    retrieved_docs = state['vector_store'].similarity_search(state["question"])
    # similarity search is not so great; maybe try again with better embedding function in the future; or split input into further keywords?
    # for now, for time-bound results, I'm going to keep going. (MAY-25)

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
    input_text = f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Question: {question} \n Context: {texts}"

    return{'question':input_text}

def generate(state:GraphState):
    question = state['question']
    tool_msg = state['tool_message']
    tool_result = state['tool_result']

    final_input = f"""
    You are answering a scientific question. \n
    User Question and Context: {question} \n
    Tool Used (if Any): {tool_msg}\n
    Tool Result (if Any): {tool_result}
    \n
    Answer the question using the above information.\n 
    Stop after reaching the answer to the question. Answer in the format: 'Final Answer: <Animal System> - <Specific Strain>.'
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

# AI Router chooses Parsing function----------------------------
'''
llm_parse_tools = llm.bind_tools([search_and_parse,just_parse])
def parsing_tool_call(state:GraphState):
    return{"messages":[llm_parse_tools.invoke(state["question"])]}

tool_parse = ToolNode(tools=[search_and_parse,just_parse])
'''
# Manual choosing of Parsing function -------------------------
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
    ['question 1',['for the micronucleus assay, is it an in vivo study or an in vitro study?',['micronucleus']]],
    ['question 2',['for the micronucleus assay, what is the animal test system used in the study? Please answer with the animal type (e.g. mouse, rat,rabbit..etc.) and species (e.g. Sprague-Dawley) if possible.',['micronucleus','strain']]]
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
# GraphState is a type, not instance.

llm.client.close()
