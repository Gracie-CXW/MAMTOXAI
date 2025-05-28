# only for testing purposes
from langchain_community.llms import LlamaCpp
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from typing_extensions import TypedDict, List, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from utils import search_and_parse, just_parse, calculator_add, calculator, max_min
import time
from typing import Annotated
from langchain_core.prompts import PromptTemplate
import os

def just_parse(pdf:str,vector_dir:str):
    import pymupdf
    import chromadb
    from langchain_chroma import Chroma
    import uuid
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    try:
      doc = pymupdf.open(pdf)
      pages = [page.get_text() for page in doc]
      Title = ''.join(pages[0].split('\n')[:10])
      text = "\n\n".join(pages)
      uuid_doc = str(uuid.uuid4())
      documents = [Document(
          page_content = text,
          metadata = {'title':Title,
          'length':len(pages)},
          id = uuid_doc
      )]

      collection_name = str(uuid.uuid4())  # Unique collection name

      # Initialize Vector Store
      embedding_func = HuggingFaceEmbeddings(model_name=r"\embeddings_local\all-MiniLM-L6-v2")
      vector_store = Chroma(
          collection_name=collection_name,
          embedding_function=embedding_func,
          persist_directory=vector_dir
      )

      # Add pages to VDB
      if pages:  # Only add if there are relevant pages
          vector_store.add_documents(
              documents = documents,
              ids = uuid)

      # For one pdf only
      return vector_store

      # For multiple PDFs, simply create global dictionary and update.
    except FileNotFoundError:
        raise ValueError(f"PDF file not found: {pdf}")
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

# Saving this chunck to test models for tool calling. 
'''
llm_tool = LlamaCpp(
    model_path = r".\models\phi-4-reasoning-plus\Phi-4-reasoning-plus-Q2_K.gguf",
    n_ctx = 32768,
    n_threads = 8,
    n_batch = 128,
    temperature = 0.5,
    verbose = False,
    #n_gpu_layers = 40 # layers to route to gpu
)

augmented_question = """
    Question:
 for the micronucleus assay, is it an in vivo study or an in vitro study? Answer ONLY in the format: Micronucleus - <in vitro/in vivo>. This question does not require any calculations.
 
    Context:
 In Vivo MAMMALIAN ERYTHROCYTE MICRONUCLEUS TEST Performed in Rat Bone Marrow Combined to THE In Vivo MAMMALIAN ALKALINE COMET ASSAY ON LUNG AND LIVER approach to genotoxicity testing in which a single study combines the analysis of micronuclei in bone that the test compound is non-genotoxic in vivo. The current study was performed in accordance with the Final Study Plan FSP-IPL 180502 (Appendix Initiated – completed : 25/06/18 – 25/09/2018 [TABLE DATA REMOVED] activity of the test item by detecting micronucleated polychromatic erythrocytes in the bone marrow of (SSB or DSB), alkali-labile sites, DNA-DNA / DNA-protein cross-linking and SSB associated with and liver) are isolated for both micronucleus test and the comet assay. 2.1. In vivo micronucleus test [TABLE DATA REMOVED] [TABLE DATA REMOVED] effect. Clastogen may produce, at the moment of mitosis, chromosome breakage, while spindle poisons disturb the structure of the mitotic spindles. An acentric fragment of a chromosome that has not migrated normally is not retained in the nucleus of the daughter cell and appears in the cytoplasm. It is then and the micronucleus remains in the erythrocytes.

[TABLE DATA REMOVED] The potential clastogenic activity of ____________ (batch 18030702) provided by _________ was tested using the in vivo micronucleus test in female rat, in compliance with the OECD Guideline No. 474, by in the frequencies of micronucleated polychromatic erythrocytes was found in the animals treated with Single cell preparation was done within one hour after animal sacrifice. A 'V' shaped incision was made from the centre of the lower abdomen to the rib cage. The skin and muscles was removed to reveal the abdominal cavity. A portion of the lung and liver was removed and washed in the cold mincing buffer until as much blood as 9.1.2. Preparation of specific reagents for comet assay [TABLE DATA REMOVED] [TABLE DATA REMOVED] dissolved at either 0.8 or 1.5% (w/v) in phosphate buffer (Ca++, Mg++ free and phenol free) by heating in a microwave. [TABLE DATA REMOVED] in phosphate buffer (Ca++, Mg++ free and phenol free) by heating in a microwave. During the study this solution was kept at 37-45°C and discarded afterward.

The test item (batch 18030702), provided by _______, was investigated for genotoxic potential by [TABLE DATA REMOVED] under alkaline conditions (SCGE) in the Lung and Liver, in female OFA Sprague-Dawley rats, according to OECD Guidelines (Nos. 474 and 489, 2016). Animals were treated endotracheally at labile sites inducer activities toward the lung and liver from OFA Sprague-Dawley female rats. Furthermore, ____________ induced no genotoxic activity in bone marrow cells.

Appendix No. 4c: Historical data for the micronucleus assay PCE/NCE ratio for 1000 PCE 1.02 [TABLE DATA REMOVED] [TABLE DATA REMOVED] Frequency of micronuclei for 1000 PCE 0.69 from May 2011 to October 2017 (20 assays)"""

question = """for the micronucleus assay, is it an in vivo study or an in vitro study? Answer ONLY in the format: Micronucleus - <in vitro/in vivo>. This question does not require any calculations."""

template = f"""
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
call_init = llm_tool.invoke(template).strip()
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
    \n{augmented_question}
    """
    call_result = llm_tool.invoke(template2).strip()
else:
    call_result = 'no tools needed'

print(f'Here is the tool call result: {call_result}')
'''

# Somebody put me out of my misery
