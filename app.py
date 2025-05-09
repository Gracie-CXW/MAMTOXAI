# In progress. 
# As of MAY-09-2025: Cleaned up initialization, removed messy and errornoeus sections. Developing tools for the AI. AI flow constructed. 

# This code is the core executable file for the MAMTOX AI 
# The model chosen for initial demonstration of the project is DeepSeek R1 1.5B, imported from huggingface 
# Ctrl + I for help

import streamlit as st 
import lanchain 
from langchain.llms import Llamacpp
from langchain.chains import RetrievalQA
from langchain_core.messages import AIMessage, HumanMessage
from utils import empty_cache
import re 
import os 
import sys 
import chromadb 
from chromadb.configs import settings 
import shutil

# Initialization 
pdf = st.file_uploader("Upload your toxicity report (must be PDF)",type=['pdf'])
database = st.file_uploader("Optional: Upload a pre-filled database for the model to base its predictions on",type=['csv','excel'])
query = st.text_input("Start by asking DATA a question!")

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

# Construct the Graph AI -----------------------------------------------------------------------------------------------------------------------------------------
import langgraph
from langgraph.graph import StateGraph, START, END 
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition



# Bind tools from .utils to the llm
tools_llm = llm.bind_tools([]) # put functions here

def llm_tooL_node(state: MessagesState):
    return{"messages":[tools_llm.invoke(state["messages"])]}
    
