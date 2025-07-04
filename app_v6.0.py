# MAY-23-2025-FRIDAY==========================================================================================
from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing_extensions import TypedDict, List, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from utils import search_and_parse, just_parse, calculator_add, calculator, max_min,cleanup_doc,empty_cache,load_saved_embeds,save_vector_store
import time
from typing import Annotated
from langchain_core.prompts import PromptTemplate
import os
import pymupdf
import chromadb
from langchain_chroma import Chroma
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
# Last Updated: JUN-12, 12:29pm

# Define the local QNA model -------------------------------------------------------------------------------------
llm = ChatOllama(
    model = 'phi4',
    num_ctx = 16000,
    temperature = 0.7,
    verbose = False,
    num_gpu = 1, # number of gpus to use
    num_predict = 500,
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
    vector_store_infos: any # technically list of list of strings, but i put any so that initialization doesnt cause any errors.
    tool_message: str
    pdf_fp:str
    vectordb_dir:str 
    cache_dir:str
    vector_save_fp:str
    corrected_output:str
    final_input:str

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
    embedding_func = HuggingFaceEmbeddings(model_name=r".\embeddings_local\all-MiniLM-L6-v2")
    vb_infos = state['vector_store_infos']

    if state['vector_store_infos'] is None:
        raise ValueError("Vector store(s) not initialized!")
    
    retrieved_docs = []
    for vb_info in vb_infos:
        try: # Only retrieving documents with ids (successful searches)
            unique_ids = set(vb_info[2])
            vector_store = Chroma(
                persist_directory = vb_info[0],
                embedding_function = embedding_func,
                collection_name = vb_info[1]
            )
            docs = vector_store.get_by_ids(unique_ids) # List[doc]
            for n_doc in docs:
                retrieved_docs.append(n_doc)
        except Exception as e:
            print(f"Error occured in retrieving one of the context vector bases: {e}.")
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
    return{
        'output':output,
        'final_input':final_input
    }

def formatter(state:GraphState):
    output = state['output'].text()
    template = f"""
        You are an scientific assistant, your task is to read a response and correctly format the answer.\n
        The Response is as follows:\n
        {output}\n

        Formatting Rules are as follows:\n 
        - Disgard every thing aside from the answer, this includes all thinking processes or justifications for the answer.\n 
        - Ensure that the final response contains ONLY lines in this EXACT format: <value 1> : <value 2>.\n
        \n 
        Acceptable Ouptut Examples:
        DERMAL : Sensitization \n 
        PURITY : 93.4% \n 
        SAMPLE NUMBER : 45\n
        \n
        """

    corrected_output = llm.invoke(template)
    conversation_history = [
        HumanMessage(content=state['final_input']),
        corrected_output
    ]

    return {
        'corrected_output':corrected_output,
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
    vector_set = load_saved_embeds(state['vector_save_fp']) # i need to change the name bc this is not a set it is a list of lists...
    pdf_fp = state['pdf_fp']
    keywords = state['keywords']
    vb_infos = []
    cache_dir = state['cache_dir']

    pdf_name = os.path.basename(pdf_fp).split('.')
    pdf_name.remove('pdf')
    pdf_name = "".join(pdf_name)

    existing_embeddings = os.listdir(state['vectordb_dir'])
    # First check if vector store already exists
    for k in keywords:
        k_name = "_".join(k)
        modified_name = f"{pdf_name}_{k_name}_searchandparse"
        persist_directory = os.path.join(state['vectordb_dir'],modified_name)

        if modified_name in existing_embeddings: # if folder exists
            for vb_info in vector_set:
                if vb_info[0] == persist_directory: # if saved file exists
                    vb_infos.append(vb_info)
                    print(f"SnP: PDF: '{pdf_name}' has been previously searched and parsed and found in directory '{persist_directory}'. \nThe previous embedding will be used if not empty.")
            if not any(vb_info[0] == persist_directory for vb_info in vector_set):
                print(f"SnP: PDF: '{pdf_name}' embedding directory is initialized in the path: '{persist_directory}' but it is empty or incomplete. \nA new embedding will be made for this PDF within the directory '{persist_directory}'.")
                vb_info = search_and_parse(pdf_fp,k,state['vectordb_dir'],cache_dir,state['vector_save_fp'])
                vb_infos.append(vb_info)
                
        else: # if folder does not exist
            print(f"SnP: PDF: '{pdf_name}' new to search_and_parse under the specified functions: \n{k},\na new embedding will be made for it within the directory '{persist_directory}'.")
            vb_info = search_and_parse(pdf_fp,k,state['vectordb_dir'],cache_dir,state['vector_save_fp'])
            vb_infos.append(vb_info)
    
    return {'vector_store_infos':vb_infos}
                
def parse_all(state:GraphState):
    pdf_fp = state['pdf_fp']
    vector_set = load_saved_embeds(state['vector_save_fp']) # i need to change the name bc this is not a set it is a list of lists...
    pdf_name = os.path.basename(pdf_fp).split('.')
    pdf_name.remove('pdf')
    pdf_name = "".join(pdf_name)
    modified_name = f"{pdf_name}_justparse"
    persist_directory = os.path.join(state['vectordb_dir'],modified_name)
    vb_infos = []

    existing_embeddings = os.listdir(state['vectordb_dir'])
    # First check if vector stroe already exists
    if modified_name in existing_embeddings:
        for vb_info in vector_set:
            if vb_info[0] == persist_directory:
                vb_infos.append(vb_info)
                print (f"JP: PDF: '{pdf_name}' previously parsed via just_parse function. The previous embedding will be used. ")
        if not any(vb_info[0] == persist_directory for vb_info in vector_set):
            print(f"JP: PDF: '{pdf_name}' embedding directory is initialized in the path: '{persist_directory}' but it is empty or incomplete. \nA new embedding will be made for this PDF within the directory '{persist_directory}'.")
            vb_info = just_parse(pdf_fp,state['vectordb_dir'],state['vector_save_fp'])
            vb_infos.append(vb_info)
    
    else:
        print(f"JP: PDF: '{pdf_name}' new to just_parse.\nA new embedding will be made for it within the directory '{persist_directory}'. ")
        vb_info = just_parse(pdf_fp,state['vectordb_dir'],state['vector_save_fp'])
        vb_infos.append(vb_info)
    
    return {'vector_store_infos':vb_infos}

    
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
builder.add_node('formatter',formatter)

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
builder.add_edge('generate_1', 'formatter')
builder.add_edge('formatter',END)
graph = builder.compile()

#-----------------------------------------------Run Graph ------------------------------------------------
# Initialization --------------------------------------------------------------------------------------------
#pdf_fp = r".\pdf\IA - Proof of concept (Studies)\AI - Proof of concept (Study 1).pdf" # put pdf here
vectordb_dir = r'.\vector_db'
cache_dir = r'.\cache'
chats_dir = r".\chats"
# Set up vector store dictionary --------------------------------------------------------------------------------------
vector_save_fp = r".\embedding_dict.txt"
# Set up counter ----------------------------------------------------------------------------------------
time_per_trial = []
time_per_run = [] # for multiple runs (multiple pdfs)

# Set up questions --------------------------------------------------------------------------------------
"""
NOTICE:
Please keep in mind if you wish to perform Search and Parse on the PDFs:
The acceptable input format for the keywords list is:
[Inclusion method, Search Type, keywords] for every search.

The Inclusion method can take on the following values (case insensitive, must be strings):
-OR
-AND
-XOR
-XAND

The Search Type can take on the following values (case insensitive, must be strings):
-Literal -> searches the pdf for the literal keywords; e.g. 'genotoxicity' is not the same as 'mutagenicity'. 
-Similar -> searches the pdf for similar words to the keywords including the actual words.

You are allowed to input multiple search methods for one question, each as a list separated by commas. The search and parse function will return all matched pages.

Example inputs:
-[['XOR','Literal','genotoxicity','rat','rats'] , ['AND','similar','sensitization','dermal']]
-[['OR','similar','sensitization studies with rabbit test systems.'],['OR','literal','rabbit','patch']]

IMPORTANT: Similarity searches are more leniant with the input type, therefore you can input short sentences such as 'the genotoxicity study with rats as the test system'! Literal seraches are more strict, so stick to single words!
"""

# input format example ['question 1',['what is...?',['micronucles','in vivo']]]
inputs = [
    ['question 1',[
        """
            Determine the exposure type (ORAL, DERMAL, or INHALATION) for the toxicology study, then classify the exposure method based on these rules:\n
            -DERMAL study exposure methods: Topical Application, Intradermal Injection, or Occlusive Patch \n
            -ORAL study exposure methods: gavage or feed  \n
            -INHALATION study exposure methods: Powder, Vapor, or Gas Chamber\n
            \n
            Format your answer exactly as: <EXPOSURE TYPE>: <EXPOSURE METHOD>  \n
            If either is missing or unclear, return: Null\n
            Examples of acceptable answers:\n
            - DERMAL: Topical Application  \n
            - ORAL: gavage  \n
            - Null : Null \n
            IMPORTANT: Ensure the last line of your output is the answer in the exact format specified above.\n
            There may be multiple studies inside this report.
    """,[]
    ]],
    ['question 2',[
        """
        Find the purity of the tested substance for this toxicology report.\n
        Your answer should follow the format: PURITY : <purity%> or PURITY : no info if the purity is not mentioned.\n
        Examples of acceptable outputs:\n
        - PURITY : 92.0%\n
        - PURITY : no info\n
        IMPORTANT: Ensure the last lines of your output is the answer in the format specified above.\n
        There may be multiple correct answers.
        """,[ ['or','similar','purity'] ]]]
    ,
    ['question 3',[
        """
        Find the vehicle(s) or solvent(s) used in this toxicity study.\n
        Valid examples include: alcohol, water, methanol, DMSO, oils, aqueous methylcellulose, acetone, petrolatum, sodium chloride, gelatin capsule — but other answers are allowed.\n
        \n
        Output format (strict):\n
        - SOLVENT : <solvent> (e.g., SOLVENT : DMSO)\n
        - SOLVENT : no info (if not found or unclear)\n
        Examples of acceptable outputs:\n
        - SOLVENT : DMSO\n
        - SOLVENT : no info\n
        IMPORTANT: Ensure the last line(s) of your output is(are) the answer in the format specified above.\n
        There may be multiple correct answers.
        """,[ ['or','similar','vehicle','solvent','dissolved in','prepared in'] ]]
    ]]

# Put the folder path containing all of your PDFs here, and indicate the run number (for duplicate trials) ----------------------------------------------------------------------------------------------------------
pdf_dir = r".\pdf\IA - Proof of concept (Studies)"
run_number = 1

all_studies = os.listdir(pdf_dir)
for idx_s, study in enumerate(all_studies):
    study_fp = os.path.join(pdf_dir,study)
    run_store_name = f"{study}_run_{str(run_number)}_full_convo.txt"
    response_store_name = f"{study}_run_{str(run_number)}_response_only.txt"
    start_time_s = time.time()
    with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as init_r:
        init_r.write(f"Complete Message History for PDF [{study}]")
    with open(os.path.join(chats_dir,response_store_name),'a',encoding ='utf-8') as init:
        init.write(f"Responses Only for PDF [{study}]")    
    for idx,q in enumerate(inputs):
        question_idx = q[0]
        question = q[1][0]
        keywords = q[1][1]
        start_time = time.time()
        response = graph.invoke({
            'question':question,
            'augmented_question':None,
            'keywords':keywords,
            'context':[],
            'tool_result':None,
            'output':None,
            'messages':[],
            'tool_message':None,
            'pdf_fp':study_fp,
            'vectordb_dir':vectordb_dir,
            'cache_dir':cache_dir,
            'vector_save_fp':vector_save_fp,
            'vector_store_infos':[],
            'corrected_output':None
        })
        with open(os.path.join(chats_dir,run_store_name),'a',encoding='utf-8') as run:
            records = [message.pretty_repr() for message in response['messages']]
            run.write(f"\nConversation for Question {idx+1}"+"--------------------------------------------------------------------------------------------------\n")
            for r in records:
                run.write(r + '\n')

        with open(os.path.join(chats_dir,response_store_name),'a',encoding='utf-8') as resp:
            resp.write(f"\nResponse for Question {idx+1}" + "-------------------------------------------------------------------------------------------------\n")
            resp.write(response['corrected_output'].pretty_repr())
        
        end_time = time.time()
        duration = (end_time-start_time)/60
        time_per_trial.append(duration)
        print(f'Question {idx+1} of Study {idx_s+1} complete, time took: {duration:.2f} minutes.')
        try:
            llm.client.close()
            llm_tool.client.close()
        except Exception as e:
            pass
    end_time_s = time.time()
    duration_s = (end_time_s - start_time_s)/60
    print(f"Study {idx_s+1} complete.\n timme took:{duration_s:.2f} minutes. Questions asked: {len(inputs)}")

    # GraphState is a type, not instance.