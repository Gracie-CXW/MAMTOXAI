# MAY-23-2025-FRIDAY==========================================================================================
from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing_extensions import TypedDict, List, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from utils import search_and_parse, just_parse, calculator_add, calculator, max_min,cleanup_doc,empty_cache,load_dict_from_txt,save_vector_store
import time
from typing import Annotated
from langchain_core.prompts import PromptTemplate
import os
import pymupdf
import chromadb
from langchain_chroma import Chroma
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import ChatOllama
# Last Updated: MAY-29, 3:12am

# Define the local QNA model -------------------------------------------------------------------------------------
llm = ChatOllama(
    model = 'phi4',
    num_ctx = 16000
    temperature = 0.7,
    verbose = False,
    num_gpu = , # layers to route to gpu
    num_predict = 250
    repeat_penalty = 1.2,
    top_k = 50,
    top_p = 0.85
)
# Define the local Tool Calling model ---------------------------------------------------------------------------
llm_tool = LlamaCpp(
    model_path = r".\models\phi-4-reasoning-plus\Phi-4-reasoning-plus-Q2_K.gguf",
    n_ctx = 32768,
    n_batch = 512,
    temperature = 0.5,
    verbose = False,
    n_gpu_layers = -1,
    repeat_penalty = 1.2
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
    pdf_fp:str
    vectordb_dir:str 
    cache_dir:str
    vector_save_fp:str
    ids:List[str] # Vector Store Document IDs
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
    ids = state['ids']
    if state['vector_store'] is None:
        raise ValueError("Vector store not initialized")
    #retrieved_docs = state['vector_store'].similarity_search(state["question"])
    if len(ids)>0:
        retrieved_docs = state['vector_store'].get_by_ids(ids)
    else:
        retrieved_docs = []

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
    if len(state['context']) > 0:
        docs = cleanup_doc(state['context']) # [Document,Document...]
        texts = '\n\n'.join(doc.page_content for doc in docs)

        question = state['question']
        input_text = f"""
        Below are relevant excerpts from the toxicology report that may contain the answer to the question:\n
        The excerpts are in raw text format, and may not make sense entirely (may begin or end in the middle of a sentence).\n
        If a section is irrelevant, nonsensical, or does not help answer the question, ignore it.\n
        -------------------BEGIN EXERPT-------------------n
        {texts}\n
        -------------------END EXERPT-------------------n
        Use ONLY the above information AND the Calculation Results (if any) to answer The Question:\n {question}\n
        """
    else:
        input_text = f"""
        Question:\n {state['question']}\n 
        IMPORTANT: There is no information found on the toxicology report that may provide an answer to the question. This question has no answer.
        """

    return{'augmented_question':input_text}

def generate(state:GraphState):
    question = state['augmented_question']
    tool_msg = state['tool_message']
    tool_result = state['tool_result']

    final_input = f"""
    You are an assistant tasked with extracting data from a mammalian toxicology report. You are being asked a question which you must answer.
    {question}
    \nCalculations Used in answering question(if Any): {tool_msg}
    \nCalculation Result (if Any): {tool_result}
    \n
    \nYOU MAY NOW WRITE YOUR ANSWER, STOP GENERATING after you've answered the question, you MUST output an answer.:
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
    vector_dict = load_dict_from_txt(vector_save_fp)
    pdf_fp = state['pdf_fp']
    pdf_name = os.path.basename(state['pdf_fp'])
    keywords = state['keywords']
    vector_store = state['vector_store']
    
    # First check if vector store already exists
    pdf_name = pdf_name.split('.')
    pdf_name.remove('pdf')
    pdf_name = "".join(pdf_name)
    keywords_name = "_".join(keywords)
    modified_name = f"{pdf_name}_{keywords_name}_searchandparse"
    persist_dir = os.path.join(vectordb_dir,modified_name)

    keys = list(filter(lambda key: vector_dict[key][0] == pdf_name, vector_dict))

    if len(keys) < 1: # No matching pdf to vector store with snp function
        print(f"Parsing Note (SnP): No Instances Found: I will now create a new embedding for pdf '{pdf_name}' within the directory '{persist_dir}'.")
        vector_store,persist_directory,collection_name,ids = search_and_parse(pdf_fp,keywords,vectordb_dir)
        save_vector_store(
            persist_directory = persist_directory,
            original_doc = pdf_name,
            collection_name = collection_name,
            ids=ids,
            vector_save_fp = vector_save_fp
        )

    else: # at least one occurance of pdf matched to a vector store with snp function 
        if persist_dir in os.listdir(vectordb_dir): # matching embedding with same persist_directory found (includes keywords)
            print(f"Parsing Note (SnP): At Least One Instance Found: I will now load the previously saved embedding for pdf '{pdf_name}'.")
            embedding_func = HuggingFaceEmbeddings(model_name=r".\embeddings_local\all-MiniLM-L6-v2")
                
            for k in keys:
                if k == persist_dir:
                    collection_name = vector_dict.get(k)[1]
                    vector_store = Chroma(
                        persist_directory = k,
                        embedding_function = embedding_func,
                        collection_name = collection_name
                    )
                    ids = vector_dict.get(k)[2]
                else:
                    print(f'Error: Vector store exists for pdf "{pdf_name}", but no directory can be found in save file corresponding to "{persist_dir}"')
                    vector_store,persist_directory,collection_name,ids = search_and_parse(pdf_fp,state['keywords'],vectordb_dir)
                    save_vector_store(
                    persist_directory = persist_directory,
                    original_doc = pdf_name,
                    collection_name = collection_name,
                    ids = ids,
                    vector_save_fp = vector_save_fp
                    )
        else: # no matching embedding with same persist_directory found (includes keywords)
            print(f"Parsing Note (SnP): No Instances Found: I will now create a new embedding for pdf '{pdf_name}' within the directory '{persist_dir}'")
            vector_store,persist_directory,collection_name,ids = search_and_parse(pdf_fp,state['keywords'],vectordb_dir)
            save_vector_store(
            persist_directory = persist_directory,
            original_doc = pdf_name,
            collection_name = collection_name,
            ids = ids,
            vector_save_fp = vector_save_fp
            )
    return {'vector_store':vector_store,'ids':ids}
    #return {'vector_stores': {pdf_fp:vector_store}} # for multiple pdfs
                
def parse_all(state:GraphState):
    pdf_fp = state['pdf_fp']
    vector_dict = load_dict_from_txt(vector_save_fp)
    pdf_name = os.path.basename(state['pdf_fp'])
    vector_store = state['vector_store']

    # First check if vector store already exists
    pdf_name = pdf_name.split('.')
    pdf_name.remove('pdf')
    pdf_name = "".join(pdf_name)
    modified_name = f"{pdf_name}_justparse"
    persist_dir = os.path.join(vectordb_dir,modified_name)

    keys = list(filter(lambda key: vector_dict[key][0] == pdf_name, vector_dict))

    if len(keys) < 1: # No matching pdf to vector store with jp function
        print(f"Parsing Note (JP): No Instances Found: I will now create a new embedding for pdf '{pdf_name}' within the directory '{persist_dir}'.")
        vector_store,persist_directory,collection_name,ids = just_parse(pdf_fp,vectordb_dir)
        save_vector_store(
            persist_directory = persist_directory,
            original_doc = pdf_name,
            collection_name = collection_name,
            ids=ids,
            vector_save_fp = vector_save_fp
        )

    else: # at least one occurance of pdf matched to a vector store with jp function
        print(f"Parsing Note (JP): At Least One Instance Found: I will now load the previously saved embedding for pdf '{pdf_name}'.")
        persist_directory = os.path.join(vectordb_dir,modified_name)
        embedding_func = HuggingFaceEmbeddings(model_name=r".\embeddings_local\all-MiniLM-L6-v2")
        
        for k in keys:
            if k == persist_directory:
                collection_name = vector_dict.get(k)[1]
                vector_store = Chroma(
                    persist_directory = k,
                    embedding_function = embedding_func,
                    collection_name = collection_name
                )
                ids = vector_dict.get(k)[2]
            else:
                print(f'Error: Vector store exists for pdf "{pdf_name}" but not saved in save file.')
                vector_store,persist_directory,collection_name,ids = just_parse(pdf_fp,vectordb_dir)
                save_vector_store(
                    persist_directory = persist_directory,
                    original_doc = pdf_name,
                    collection_name = collection_name,
                    ids=ids,
                    vector_save_fp = vector_save_fp
                )

    return{'vector_store':vector_store,'ids':ids}
    #return {'vector_stores': {pdf_fp:vector_store}} # for multiple pdfs

def snp_condition(state:GraphState):
    if state['keywords'] and any(k is not None for k in state['keywords']):
        return 'search_and_parse'
    return 'just_parse'

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
# Initialization --------------------------------------------------------------------------------------------
#pdf_fp = r".\pdf\IA - Proof of concept (Studies)\AI - Proof of concept (Study 1).pdf" # put pdf here
vectordb_dir = r'.\vector_db'
cache_dir = r'.\cache'
# Set up vector store dictionary --------------------------------------------------------------------------------------
vector_save_fp = r".\embedding_dict.txt"
vector_dict = {}
# Set up counter ----------------------------------------------------------------------------------------
time_per_trial = []
time_per_run = [] # for multiple runs (multiple pdfs)
# Set up questions --------------------------------------------------------------------------------------
inputs = [
    ['question 1',[
        """ Determine the exposure type (ORAL, DERMAL, or INHALATION) for the toxicology study, then classify the exposure method based on these rules:\n
            -DERMAL study exposure methods: Topical Application, Intradermal Injection, or Occlusive Patch \n
            -ORAL study exposure methods: gavage or feed  \n
            -INHALATION study exposure methods: Powder, Vapor, or Gas Chamber\n
            \n
            Format your answer exactly as: <EXPOSURE TYPE>: <EXPOSURE METHOD>  \n
            If either is missing or unclear, return: Null\n
            Examples of acceptable answers:\n
            - DERMAL: Topical Application  \n
            - ORAL: gavage  \n
            - Null\n""",[]
    ]],
    ['question 2',[
        """Find the purity of the tested substance for this toxicology report.\n
        Your answer should follow the format: PURITY : <purity%> or PURITY : no info if the purity is not mentioned.\n
        Ensure the last line of your output is the answer in the format specified above.\n
        Examples of acceptable outputs:\n
        - PURITY : 92%\n
        - PURITY : no info\n
        """,['purity']
    ]],
    ['question 3',[
        """Find the vehicle(s) or solvent(s) used in this toxicity study.\n
        Valid examples include: alcohol, water, methanol, DMSO, oils, aqueous methylcellulose, acetone, petrolatum, sodium chloride, gelatin capsule — but other answers are allowed.\n
        \n
        Output format (strict):\n
        - SOLVENT : <solvent> (e.g., SOLVENT : DMSO)\n
        - SOLVENT : no info (if not found or unclear)\n
        Examples of acceptable outputs:\n
        - SOLVENT : DMSO\n
        - SOLVENT : no info\n
        """,[]
    ]]
]
# format example ['question 1',['what is...?',['micronucles','in vivo']]]
# question:str, keywords:list()

pdf_dir = r".\pdf\trial-1-test-focus"
all_studies = os.listdir(pdf_dir)

for idx_s, study in enumerate(all_studies):
    study_fp = os.path.join(pdf_dir,study)
    run_store_name = f"run_{str(idx_s+1)}_04.txt"
    start_time_s = time.time()
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
            'tool_message':None,
            'pdf_fp':study_fp,
            'vectordb_dir':vectordb_dir,
            'cache_dir':cache_dir,
            'vector_save_fp':vector_save_fp,
            'ids':[]
        })
        with open(os.path.join(cache_dir,run_store_name),'a',encoding='utf-8') as cache:
            records = [message.pretty_repr() for message in response['messages']]
            cache.write(f"Conversation for Question {idx+1}"+"===========================================================================\n")
            for r in records:
                cache.write(r + '\n')
        
        end_time = time.time()
        duration = (end_time-start_time)/60
        time_per_trial.append(duration)
        print(f'Question {idx+1} of Study {idx_s+1} complete, time took: {duration:.2f} minutes.')
    end_time_s = time.time()
    duration_s = (end_time_s - start_time_s)/60
    print(f"Study {idx_s+1} complete.\n timme took:{duration_s:.2f} minutes. Questions asked: {len(inputs)}")

    # GraphState is a type, not instance.

llm.client.close()
