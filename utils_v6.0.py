# In Progress
# As of MAY-09-2025: Written some core tools for the AI to utilize. Still yet to implement chat memeory or tools to decode files other than pdfs. Currently, only one question can be asked, and no cache is available. 
# As of MAY-15-2025: Written tool for query breakdown, created pull request for using langchain built-in chroma module instead of chromadb itself in fear of compatibility issues. Updated parsing functions, original code was messy af.
# This file contains possible tools for the model to utilize

# Last Updated: May-31, 3:41am
import sys

def empty_cache(fp:str):
    """
    Input arguments:
    fp: str. directory to clear. 

    Output:
    None

    deletes the directory at the given filepath. 
    then remakes the directory.
    """
    import shutil
    import os
    if os.path.exists(fp):
        shutil.rmtree(fp)
        os.makedirs(fp)

def save_vector_store(persist_directory:str,collection_name:str,ids:list[str],vector_save_fp:str):
    #import pickle
    # idk how to use pickle
    """
    Input Arguments:
    Persist_directory: str, filepath for the vector store embedding.
    Collection_name: str, name of vector store
    ids: list(str), list of page ids in the vector_store
    vector_save_fp: str, the filepath of the file you wish to store all of the embedding informations.
    Output:
    None
    
    Save file has following syntax: 
    persist_directory,collection_name,ids
    """
    with open(vector_save_fp,'a') as f:
        line = f'{persist_directory},{collection_name},'+','.join(ids)
        f.write(line)
        f.write('\n')

def load_saved_embeds(save_fp):
    #import pickle 
    # idk how to use pickle
    import os
    """
    Input argument:
    save_fp: str, filepath of the saved dictionary.

    Output:
    vector_infos: list[list] in the format set([persist_directory,collection_name,list(ids)])
    """
    vector_infos = []

    if not os.path.exists(save_fp):
        raise Exception(f"save file not found or cannot be accessed at the following directory: '{save_fp}'.")

    with open(save_fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                information = line.split(',')
                ids = information[2:]
                vector_infos.append([information[0],information[1],ids])
    
    return vector_infos

def just_parse(pdf:str,vector_dir:str,vector_save_fp:str):
    """
    Input Arguments: 
    PDF file path: str 
    Vector Database Directory: str
    Vector Save Filepath: str

    Outputs:
    vector_store_infos: list[str], in the format [persist_directory,collection_name,ids]
    """

    import pymupdf
    import chromadb
    from langchain_chroma import Chroma
    import uuid
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    import os 

    pdf_name = os.path.basename(pdf).split('.')
    pdf_name.remove('pdf')
    pdf_name = "".join(pdf_name)
    modified_name = f"{pdf_name}_justparse"
    
    try:
      doc = pymupdf.open(pdf)
      pages = [page.get_text() for page in doc]
      Title = ''.join(pages[0].split('\n')[:10])
      uuid_doc = str(uuid.uuid4())
      ids = [str(idx) for idx in range(len(pages))]

      documents = [Document(
          page_content = text,
          metadata = {'title':Title,
          'length':len(pages),
          'parse function':'just_parse'},
          id = str(idx)
      ) for idx,text in enumerate(pages)]

      collection_name = str(uuid.uuid4())  # Unique collection name

      # Initialize Vector Store
      embedding_func = HuggingFaceEmbeddings(model_name=r".\embeddings_local\all-MiniLM-L6-v2")
      persist_directory = os.path.join(vector_dir,modified_name)
      vector_store = Chroma(
          collection_name=collection_name,
          embedding_function=embedding_func,
          persist_directory=persist_directory # Creates folder to store vector store.
      )

      # Add pages to VDB
      if pages:  # Only add if there are relevant pages
          vector_store.add_documents(
              documents = documents,
              ids = ids)

      vector_store_infos= [persist_directory,collection_name,ids]
      save_vector_store(persist_directory,collection_name,ids,vector_save_fp)
      return vector_store_infos

    except FileNotFoundError:
        raise ValueError(f"PDF file not found: {pdf}")
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def calculator_add(*args):
    """
    Input Arguments:
    int or float.

    Output:
    int or float.

    Created as a separate function to calculator for adding animal test sample sizes, which is more commonly used for us.
    """
    # Specifically for calculating gross sample sizes
    return sum(args)

def calculator(operation,*args):
    """
    Input Arguments:
    operation: str, the mathematical operation to use.
    ints or floats.

    Output:
    int or float.

    Includes all calculations aside from addition.
    """
    operations = {
        'multiply': lambda *args: reduce(lambda x, y: x * y, args),
        'subtract': lambda x, y: x - y,
        'divide': lambda x, y: x / y,
        'power': lambda x, y: x ** y,
        'root': lambda x, y: x ** (1 / y),
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    # Validate the number of arguments based on the operation
    if operation in ['multiply']:
        if len(args) < 2:
            raise ValueError(f"{operation} requires at least two numbers")
    else:
        if len(args) != 2:
            raise ValueError(f"{operation} requires exactly two numbers")

    # Special case checks
    if operation == 'divide':
        if args[1] == 0:
            raise ValueError("Cannot divide by zero")
    elif operation == 'root':
        if args[1] == 0:
            raise ValueError("Root index cannot be zero")

    # Execute the operation and return the result
    func = operations[operation]
    return func(*args)

def search_and_parse(pdf: str, keywords: list[str], vector_dir: str,cache_dir:str,vector_save_fp:str):
    """
    Input Arguments:
    PDF file path: str
    Keywords: list of str, only matches pages with ALL keywords present. First entry is the inculsion method, second entry is the search type, rest are keywords.

    Each Search and Parse function only takes ONE search method at once!

    Allowed values for inlcusion: 
    - AND
    - OR
    - XOR
    - XAND
    Allowed values for search type:
    - Literal
    - Similar
    Vector Database Directory path: str
    Cache Directory path: str
    Vector Save File Path: str

    Outputs:
    vector_store_infos: list(str), in the format list(persist_directory,collection_name,ids)
    """
    import pymupdf
    import chromadb
    from langchain_chroma import Chroma
    import uuid
    from langchain_huggingface import HuggingFaceEmbeddings
    #from sentence_transformers import SentenceTransformer
    from langchain_core.documents import Document
    import os 

    pdf_name = os.path.basename(pdf).split('.')
    pdf_name.remove('pdf')
    pdf_name = "".join(pdf_name)
    doc = pymupdf.open(pdf)
    pages = [page.get_text() for page in doc]
    toc = doc.get_toc()

    vector_store_infos = None # List[persist_directory,collection_name,ids]

    keywords_name = "_".join(keywords)
    modified_name = f"{pdf_name}_{keywords_name}_searchandparse" # Includes inclusion methond AND search method! yay
    persist_directory = os.path.join(vector_dir,modified_name)

    inclusion_method = keywords[0].strip().lower()
    parse_method = keywords[1].strip().lower()
    keywords_lower = [k.strip().lower() for k in keywords[2:]]

    # Initialize emebedding
    # need to download embedding function bc for some proxy error for some reason.
    embedding_func = HuggingFaceEmbeddings(model_name=r".\embeddings_local\all-MiniLM-L6-v2")
    print(f"Starting search and parse function on pdf '{pdf_name}', using methods '{inclusion_method}' - '{parse_method}'.")

    # Initialize Vector Store
    collection_name = str(uuid.uuid4())  # Unique collection name
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_func,
        persist_directory=persist_directory # Creates folder to store vector store
    )

    # for the 'AND' inclusion method
    if inclusion_method == 'and':
        if parse_method == 'literal':
            try:
                relevant_pages = []
                if toc:
                    # Identify top-level sections
                    min_level = min(entry[0] for entry in toc)
                    sections = [entry for entry in toc if entry[0] == min_level]

                    # Process each section
                    for i in range(len(sections)):
                        start_page = sections[i][2] - 1  # Convert to 0-based index
                        if i < len(sections) - 1:
                            end_page = sections[i + 1][2] - 2
                        else:
                            end_page = len(pages) - 1
                        section_title = sections[i][1]
                        section_pages = list(range(start_page, end_page + 1))
                        section_texts = [pages[s] for s in section_pages]

                        combined_text = "\n".join(section_texts)

                        # Include all pages in section if any page contains all of the keywords
                        if all(k in combined_text for k in keywords_lower):
                            section_title = sections[i][1]
                            for p in section_pages:
                                relevant_pages.append([section_title, p, pages[p]])
                else:
                    # Fallback to original page-by-page approach
                    for idx, p in enumerate(pages):
                        text_lower = p.lower()
                        if all(keyword in text_lower for keyword in keywords_lower):
                            title = " ".join(p.split()[:100])
                            relevant_pages.append([title, idx, p])

                # Create Document objects
                documents = [Document(
                    page_content=text,
                    metadata={"original_doc_dir": pdf, "titles": title, "page": index,'parse function':'search_and_parse_literal'},
                    id=str(index)
                ) for title, index, text in relevant_pages]

                ids = [str(index) for title,index,text in relevant_pages]
        
                # Add pages to vector store
                if relevant_pages:
                    vector_store.add_documents(documents=documents, ids=ids)

            except FileNotFoundError:
                raise ValueError(f"PDF file not found: {pdf}")
            except Exception as e:
                raise Exception(f"Error processing PDF: {str(e)}")

        elif parse_method == 'similar':
            try: 
                full_docs = [
                    Document(
                        page_content = text,
                        metadata = {'original_doc_dir':pdf,'page':index,'parse function':'search_and_parse_similarity'},
                        id = str(index)
                    ) for index,text in enumerate(pages)
                ]
                full_vector_store = Chroma(collection_name = str(uuid.uuid4()),embedding_function = embedding_func,persist_directory=os.path.join(cache_dir,str(uuid.uuid4())))
                if full_docs:
                    full_vector_store.add_documents(full_docs)

                relevant_page_nums = []
                for k in keywords_lower:
                    relevant_k_docs = full_vector_store.similarity_search(k)
                    page_ids = [doc.id for doc in relevant_k_docs]
                    for page_id in page_ids:
                        relevant_page_nums.append(page_id)

                ids = []
                for page_num in list(set(relevant_page_nums)):
                    if relevant_page_nums.count(page_num) == len(keywords_lower):
                        ids.append(page_num)
                    elif relevant_page_nums.count(page_num) == int(len(keywords_lower)*0.75): 
                        # A more leniant qualification system includes false negatives! If there are 4 keywords and a page contains 3 of them, it is considered a relevant page.
                        ids.append(page_num)
                final_pages = full_vector_store.get_by_ids(ids)

                if final_pages:
                    vector_store.add_documents(documents=final_pages,ids=ids)
            

            except FileNotFoundError:
                raise ValueError(f"PDF file not found: {pdf}")
            except Exception as e:
                raise Exception(f"Error processing PDF: {str(e)}")

        else:
            raise Exception(f"Error during Search and Parse Function: Search method {parse_method} not recognized, please only enter one of the two acceptable methods: 'Literal' or 'Similar'. \n Enter the search method at position 1 of the keywords list.")

    # for the 'OR' inclusion method
    elif inclusion_method == 'or':
        if parse_method == 'literal':
            try:
                relevant_pages = []
                if toc:
                    # Identify top-level sections
                    min_level = min(entry[0] for entry in toc)
                    sections = [entry for entry in toc if entry[0] == min_level]

                    # Process each section
                    for i in range(len(sections)):
                        start_page = sections[i][2] - 1  # Convert to 0-based index
                        if i < len(sections) - 1:
                            end_page = sections[i + 1][2] - 2
                        else:
                            end_page = len(pages) - 1
                        section_pages = list(range(start_page, end_page + 1))
                        section_texts = [pages[s] for s in section_pages]
                        combined_text = "\n".join(section_texts)

                        # Include all pages in section if any page contains a keyword
                        if any(k in combined_text for k in keywords_lower):
                            section_title = sections[i][1]
                            for p in section_pages:
                                relevant_pages.append([section_title, p, pages[p]])
                else:
                    # Fallback to original page-by-page approach
                    for idx, p in enumerate(pages):
                        text_lower = p.lower()
                        if any(keyword in text_lower for keyword in keywords_lower):
                            title = " ".join(p.split()[:100])
                            relevant_pages.append([title, idx, p])

                # Create Document objects
                documents = [Document(
                    page_content=text,
                    metadata={"original_doc_dir": pdf, "titles": title, "page": index,'parse function':'search_and_parse_literal'},
                    id=str(index)
                ) for title, index, text in relevant_pages]

                ids = [str(index) for title,index,text in relevant_pages]

                # Add pages to vector store
                if relevant_pages:
                    vector_store.add_documents(documents=documents, ids=ids)

            except FileNotFoundError:
                raise ValueError(f"PDF file not found: {pdf}")
            except Exception as e:
                raise Exception(f"Error processing PDF: {str(e)}")

        elif parse_method == 'similar':
            try: 
                full_docs = [
                    Document(
                        page_content = text,
                        metadata = {'original_doc_dir':pdf,'page':index,'parse function':'search_and_parse_similarity'},
                        id = str(index)
                    ) for index,text in enumerate(pages)
                ]
                full_vector_store = Chroma(collection_name = str(uuid.uuid4()),embedding_function = embedding_func,persist_directory=os.path.join(cache_dir,str(uuid.uuid4())))
                if full_docs:
                    full_vector_store.add_documents(full_docs)

                relevant_ids = [] # non-unique initially
                for k in keywords_lower:
                    relevant_k_docs = full_vector_store.similarity_search(k)
                    
                    ids_k = [doc.id for doc in relevant_k_docs]
                    for ids_k_n in ids_k:
                        relevant_ids.append(ids_k_n)

                ids = set(relevant_ids)
                relevant_docs = full_vector_store.get_by_ids(ids)

                if relevant_docs:
                    vector_store.add_documents(documents=relevant_docs,ids=ids)

            except FileNotFoundError:
                raise ValueError(f"PDF file not found: {pdf}")
            except Exception as e:
                raise Exception(f"Error processing PDF: {str(e)}")
        else:
            raise Exception(f"Error during Search and Parse Function: Search method {parse_method} not recognized, please only enter one of the two acceptable methods: 'Literal' or 'Similar'. \n Enter the search method at position 1 of the keywords list.")

    # for the 'XAND' inclusion method
    elif inclusion_method == 'xand':
        if parse_method == 'literal':
            relevant_pages = []
            try:
                if toc:
                    # Identify top-level sections
                    min_level = min(entry[0] for entry in toc)
                    sections = [entry for entry in toc if entry[0] == min_level]

                    # Process each section
                    for i in range(len(sections)):
                        start_page = sections[i][2] - 1  # Convert to 0-based index
                        if i < len(sections) - 1:
                            end_page = sections[i + 1][2] - 2
                        else:
                            end_page = len(pages) - 1
                        section_title = sections[i][1]
                        section_pages = list(start_page, end_page + 1)
                        section_texts = [pages[s] for s in section_pages]

                        combined_text = "\n".join(section_texts)

                        # Exclude all pages in section if any page contains all of the keywords
                        if all(k in combined_text for k in keywords_lower):
                            pages_before = [[idx, text] in enumerate(pages[:start_page])]
                            pages_after = [[idx,text] in enumerate(pages[end_page+1:])]
                            combined_pages = pages_before+pages_after
                            for page in combined_pages:
                                relevant_pages.append(pages)
                else:
                    # Fallback to original page-by-page approach, excluding a page if it contains all keywords
                    excluded_idxs= []
                    for idx, p in enumerate(pages):
                        text_lower = p.lower()
                        if all(keyword in text_lower for keyword in keywords_lower):
                            excluded_idxs.append(idx)
                    for idx,p in enumerate(pages):
                        if idx != excluded_idxs:
                            relevant_pages.append([idx,p])

                # Create Document objects
                documents = [Document(
                    page_content=text,
                    metadata={"original_doc_dir": pdf, "page": index,'parse function':'search_and_parse_literal'},
                    id=str(index)
                ) for index, text in relevant_pages]

                ids = [str(index) for index,text in relevant_pages]

                # Add pages to vector store
                if relevant_pages:
                    vector_store.add_documents(documents=documents, ids=ids)

            except FileNotFoundError:
                raise ValueError(f"PDF file not found: {pdf}")
            except Exception as e:
                raise Exception(f"Error processing PDF: {str(e)}")

        elif parse_method == 'similar':
            try: 
                full_docs = [
                    Document(
                        page_content = text,
                        metadata = {'original_doc_dir':pdf,'page':index,'parse function':'search_and_parse_similarity'},
                        id = str(index)
                    ) for index,text in enumerate(pages)
                ]
                full_vector_store = Chroma(collection_name = str(uuid.uuid4()),embedding_function = embedding_func,persist_directory=os.path.join(cache_dir,str(uuid.uuid4())))
                if full_docs:
                    full_vector_store.add_documents(full_docs)

                excluded_idxs = []
                for k in keywords_lower:
                    relevant_k_docs = full_vector_store.similarity_search(k)
                    page_ids = [doc.id for doc in relevant_k_docs]
                    for page_id in page_ids:
                        excluded_idxs.append(page_id)

                ids = []
                for page_num in list(set(excluded_idxs)):
                    if excluded_idxs.count(page_num) == 0:
                        ids.append(page_num)
                    elif excluded_idxs.count(page_num) == int((len(keyword)-2)*0.25): 
                        # A more leniant qualification system includes false negatives! Includes pages if only 25% of the keywords are found (1 in 4).
                        ids.append(page_num)

                final_pages = full_vector_store.get_by_ids(ids)

                if final_pages:
                    vector_store.add_documents(documents=final_pages,ids=ids)

            except FileNotFoundError:
                raise ValueError(f"PDF file not found: {pdf}")
            except Exception as e:
                raise Exception(f"Error processing PDF: {str(e)}")
        else:
            raise Exception(f"Error during Search and Parse Function: Search method {parse_method} not recognized, please only enter one of the two acceptable methods: 'Literal' or 'Similar'. \n Enter the search method at position 1 of the keywords list.")

    # for the 'XOR' inclusion method
    elif inclusion_method == 'xor':
        if parse_function == 'literal':
            relevant_pages = []
            try:
                if toc:
                    # Identify top-level sections
                    min_level = min(entry[0] for entry in toc)
                    sections = [entry for entry in toc if entry[0] == min_level]

                    # Process each section
                    for i in range(len(sections)):
                        start_page = sections[i][2] - 1  # Convert to 0-based index
                        if i < len(sections) - 1:
                            end_page = sections[i + 1][2] - 2
                        else:
                            end_page = len(pages) - 1
                        section_title = sections[i][1]
                        section_pages = list(start_page, end_page + 1)
                        section_texts = [pages[s] for s in section_pages]

                        combined_text = "\n".join(section_texts)

                        # Exclude all pages in section if any page contains any of the keywords
                        if any(k in combined_text for k in keywords_lower):
                            pages_before = [[idx, text] in enumerate(pages[:start_page])]
                            pages_after = [[idx,text] in enumerate(pages[end_page+1:])]
                            combined_pages = pages_before+pages_after
                            for page in combined_pages:
                                relevant_pages.append(page)
                else:
                    # Fallback to original page-by-page approach, excluding a page if it contains all keywords
                    excluded_idxs= []
                    for idx, p in enumerate(pages):
                        text_lower = p.lower()
                        if any(keyword in text_lower for keyword in keywords_lower):
                            excluded_idxs.append(idx)
                    for idx,p in enumerate(pages):
                        if idx != excluded_idxs:
                            relevant_pages.append([idx,p])

                # Create Document objects
                documents = [Document(
                    page_content=text,
                    metadata={"original_doc_dir": pdf, "page": index,'parse function':'search_and_parse_literal'},
                    id=str(index)
                ) for index, text in relevant_pages]

                ids = [str(index) for index,text in relevant_pages]

                # Add pages to vector store
                if relevant_pages:
                    vector_store.add_documents(documents=documents, ids=ids)

            except FileNotFoundError:
                raise ValueError(f"PDF file not found: {pdf}")
            except Exception as e:
                raise Exception(f"Error processing PDF: {str(e)}")
        elif parse_function == 'similar':
            try: 
                full_docs = [
                    Document(
                        page_content = text,
                        metadata = {'original_doc_dir':pdf,'page':index,'parse function':'search_and_parse_similarity'},
                        id = str(index)
                    ) for index,text in enumerate(pages)
                ]
                full_vector_store = Chroma(collection_name = str(uuid.uuid4()),embedding_function = embedding_func,persist_directory=os.path.join(cache_dir,str(uuid.uuid4())))
                if full_docs:
                    full_vector_store.add_documents(full_docs)

                excluded_idxs = []
                ids = []
                for k in keywords_lower:
                    relevant_k_docs = full_vector_store.similarity_search(k)
                    ids_k = [doc.id for doc in relevant_k_docs]
                    for ids_k_n in ids_k:
                        excluded_idxs.append(ids_k_n)
                excluded_idxs=list(set(excluded_idxs))
                
                relevant_pages = [[idx,p] for idx,p in enumerate(pages) if idx not in excluded_idxs]
                relevant_docs = [Document(
                    page_content = text,
                    metadata = {"original_doc_dir": pdf, "page": index, "parse function": "search_and_parse_similar"},
                    id = str(index)
                ) for index,text in relevant_pages]

                if relevant_docs:
                    vector_store.add_documents(documents=relevant_docs,ids=ids)

            except FileNotFoundError:
                raise ValueError(f"PDF file not found: {pdf}")
            except Exception as e:
                raise Exception(f"Error processing PDF: {str(e)}")
        else:
            raise Exception(f"Error during Search and Parse Function: Search method {parse_method} not recognized, please only enter one of the two acceptable methods: 'Literal' or 'Similar'. \n Enter the search method at position 1 of the keywords list.")

    else:
        raise Exception(f"Error during Search and Parse Function: Inclusion method {inclusion_method} not recognized, please only enter one of the four acceptable methods: 'AND', 'OR', 'XOR', or 'XAND'.\n Enter the inclusion method at position 0 of the keywords list.")

    # Update VB information at the end 
    vector_store_infos = [persist_directory,collection_name,ids]
    save_vector_store(persist_directory,collection_name,ids,vector_save_fp)
    return vector_store_infos
  
def max_min(operation: str,*args):
    """
    Input Arguments:
    operation: str, the operation to perform; either taking the maximum or minimum of a list of numbers
    ints or floats

    Output:
    int or float
    """
    try:
        if operation == 'max':
            return max(*args)
        elif operation == 'min':
            return min(*args)
    except Exception as e:
        print (f'Error while performing max_min calculation: {str(e)}')

def cleanup_doc(pdf):
    # Removes footers, page numbers, tables, and irrelevant statistical information from texts extracted from pdf
    # !!This is not my code. It's from ChatGPT.!!
    import re
    from typing import List
    from langchain_core.documents import Document
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    def preprocess_documents(documents: List[Document]) -> List[Document]:
        """
        Preprocess LangChain Document objects to remove headers, footers, tables, and statistical data.
        
        Args:
            documents: List of LangChain Document objects, each containing page_content and metadata.
        
        Returns:
            List of modified Document objects with cleaned page_content, retaining original metadata.
        """
        # Patterns for headers and footers
        header_patterns = [
            r"^\s*(Appendix No\. \d+):.*",
            r"^\s*Table \d+.*",
            r"^\s*Figure \d+.*"
        ]
        footer_patterns = [
            r"Page \d+ of \d+",
            r"^\s*\d+\s*$"  # Standalone page numbers
        ]
        
        # Enhanced patterns for table and statistical data detection
        table_keywords = {
            "SD", "SEM", "N.S.", "p<", "p>", "p=", "t-test", "ANOVA",
            "±", "SE", "STD", "STDEV", "standard deviation", "standard error",
            "statistical", "TABLE"
        }
        
        table_patterns = [
            r"^\s*(\d+\.\d+\s+){4,}",  # Multiple decimal numbers
            r"^\s*(\d+\s+){4,}",       # Multiple whole numbers
            #r"^\s*\w+\s+\d+\s+\d+",    # Word followed by numbers (e.g., "Group 1 23 45")
            #r"^\s*[A-Z][a-z]+\s+\d+",  # Capitalized word followed by numbers
            #r"^\s*\d+\s*-\s*\d+",      # Number ranges
            #r"^\s*\d+\s*/\s*\d+"       # Number ratios
        ]
        
        # Patterns for statistical results
        stat_patterns = [
            r"p\s*[<>]?\s*0\.\d+",     # p-values
            r"\d+\s*±\s*\d+",          # Mean ± SD
            #r"\d+\s*\(\s*\d+\s*\)",    # Numbers in parentheses
            r"[A-Za-z]+\s*=\s*\d+",    # Statistical notation (e.g., "F = 23")
            r"r\s*=\s*-?\d+\.\d+"      # Correlation coefficients
        ]
        
        cleaned_documents = []
        
        for doc in documents:
            try:
                # Split page_content into lines
                lines = doc.page_content.split("\n")
                cleaned_lines = []
                in_table = False
                table_lines = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Skip headers
                    if any(re.match(pattern, line) for pattern in header_patterns):
                        logger.debug(f"Skipping header: {line}")
                        continue
                    
                    # Skip footers
                    if any(re.match(pattern, line) for pattern in footer_patterns):
                        logger.debug(f"Skipping footer: {line}")
                        continue
                    
                    # Skip lines that are entirely uppercase (often section headers or table headers)
                    if line.isupper() and len(line) > 3:
                        logger.debug(f"Skipping uppercase line (likely header): {line}")
                        continue
                    
                    # Detect table and statistical data
                    is_table_line = False
                    
                    # Check for table patterns
                    if any(re.match(pattern, line) for pattern in table_patterns):
                        is_table_line = True
                    
                    # Check for statistical keywords
                    if any(keyword.lower() in line.lower() for keyword in table_keywords):
                        is_table_line = True
                    
                    # Check for statistical patterns
                    if any(re.search(pattern, line) for pattern in stat_patterns):
                        is_table_line = True
                    
                    # Additional checks for tables
                    if line.count('\t') > 2 or line.count('  ') > 3:  # Multiple tabs or spaces
                        is_table_line = True
                    
                    if is_table_line:
                        in_table = True
                        table_lines += 1
                        logger.debug(f"Skipping table/statistical line: {line}")
                        continue
                    
                    # Reset table flag if we encounter a narrative line
                    if in_table and (len(line.split()) > 5 or len(line) > 60):
                        if table_lines < 3:  # Small tables might be false positives
                            #cleaned_lines.extend([""] * table_lines)
                            pass
                        in_table = False
                        table_lines = 0
                    
                    # Keep non-table narrative text
                    if not in_table:
                        # Remove parenthetical statistical references
                        line = re.sub(r"\([^)]*\b(p|SD|SE|SEM|n)\b[^)]*\)", "", line)
                        line = re.sub(r"\s+", " ", line).strip()
                        if line:
                            cleaned_lines.append(line)
                
                # Create new Document with cleaned content
                cleaned_content = "\n".join(cleaned_lines).strip()
                
                # Remove remaining statistical references
                cleaned_content = re.sub(r"\b(SD|SE|SEM)\b\s*[<>]?=\s*\d+\.?\d*", "", cleaned_content)
                cleaned_content = re.sub(r"\d+\s*±\s*\d+", "", cleaned_content)
                cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()
                
                if cleaned_content:
                    cleaned_doc = Document(page_content=cleaned_content, metadata=doc.metadata)
                    cleaned_documents.append(cleaned_doc)
                else:
                    logger.debug(f"Skipping empty document after cleaning (original metadata: {doc.metadata})")
                    cleaned_documents.append(doc)  # Returns original if empty after cleaning

            except Exception as e:
                logger.error(f"Error processing document with metadata {doc.metadata}: {str(e)}")
                cleaned_documents.append(doc)  # Returns original on error
                continue
        
        logger.info(f"Processed {len(documents)} documents, returned {len(cleaned_documents)} cleaned documents")
        return cleaned_documents

    clean_docs = preprocess_documents(pdf)
    return clean_docs

    # Removes footers, page numbers, tables, and irrelevant statistical information from texts extracted from pdf
    import re
    from typing import List
    from langchain_core.documents import Document
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    def preprocess_documents(documents: List[Document]) -> List[Document]:
        """
        Preprocess LangChain Document objects to remove headers, footers, tables, and statistical data.
        
        Args:
            documents: List of LangChain Document objects, each containing page_content and metadata.
        
        Returns:
            List of modified Document objects with cleaned page_content, retaining original metadata.
        """
        # Patterns for headers and footers
        header_patterns = [
            r"^\s*(Appendix No\. \d+):.*",
            r"^\s*Table \d+.*",
            r"^\s*Figure \d+.*"
        ]
        footer_patterns = [
            r"Page \d+ of \d+",
            r"^\s*\d+\s*$"  # Standalone page numbers
        ]
        
        # Enhanced patterns for table and statistical data detection
        table_keywords = {
            "MEAN", "SD", "SEM", "N.S.", "p<", "p>", "p=", "t-test", "ANOVA",
            "±", "SE", "STD", "STDEV", "standard deviation", "standard error",
            "statistical", "significant", "significance", "TABLE", "RESULT",
            "GROUP", "CONTROL", "TREATMENT", "DOSE", "mg/kg", "mL/kg"
        }
        
        table_patterns = [
            r"^\s*(\d+\.\d+\s+){2,}",  # Multiple decimal numbers
            r"^\s*(\d+\s+){3,}",       # Multiple whole numbers
            r"^\s*\w+\s+\d+\s+\d+",    # Word followed by numbers (e.g., "Group 1 23 45")
            r"^\s*[A-Z][a-z]+\s+\d+",  # Capitalized word followed by numbers
            r"^\s*\d+\s*-\s*\d+",      # Number ranges
            r"^\s*\d+\s*/\s*\d+"       # Number ratios
        ]
        
        # Patterns for statistical results
        stat_patterns = [
            r"p\s*[<>]?\s*0\.\d+",     # p-values
            r"\d+\s*±\s*\d+",          # Mean ± SD
            r"\d+\s*\(\s*\d+\s*\)",    # Numbers in parentheses
            r"[A-Za-z]+\s*=\s*\d+",    # Statistical notation (e.g., "F = 23")
            r"r\s*=\s*-?\d+\.\d+"      # Correlation coefficients
        ]
        
        cleaned_documents = []
        
        for doc in documents:
            try:
                # Split page_content into lines
                lines = doc.page_content.split("\n")
                cleaned_lines = []
                in_table = False
                table_lines = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Skip headers
                    if any(re.match(pattern, line) for pattern in header_patterns):
                        logger.debug(f"Skipping header: {line}")
                        continue
                    
                    # Skip footers
                    if any(re.match(pattern, line) for pattern in footer_patterns):
                        logger.debug(f"Skipping footer: {line}")
                        continue
                    
                    # Skip lines that are entirely uppercase (often section headers or table headers)
                    if line.isupper() and len(line) > 3:
                        logger.debug(f"Skipping uppercase line (likely header): {line}")
                        continue
                    
                    # Detect table and statistical data
                    is_table_line = False
                    
                    # Check for table patterns
                    if any(re.match(pattern, line) for pattern in table_patterns):
                        is_table_line = True
                    
                    # Check for statistical keywords
                    if any(keyword.lower() in line.lower() for keyword in table_keywords):
                        is_table_line = True
                    
                    # Check for statistical patterns
                    if any(re.search(pattern, line) for pattern in stat_patterns):
                        is_table_line = True
                    
                    # Additional checks for tables
                    if line.count('\t') > 2 or line.count('  ') > 3:  # Multiple tabs or spaces
                        is_table_line = True
                    
                    if is_table_line:
                        in_table = True
                        table_lines += 1
                        logger.debug(f"Skipping table/statistical line: {line}")
                        continue
                    
                    # Reset table flag if we encounter a narrative line
                    if in_table and (len(line.split()) > 5 or len(line) > 60):
                        if table_lines < 3:  # Small tables might be false positives
                            cleaned_lines.extend([""] * table_lines)
                        in_table = False
                        table_lines = 0
                    
                    # Keep non-table narrative text
                    if not in_table:
                        # Remove parenthetical statistical references
                        line = re.sub(r"\([^)]*\b(p|SD|SE|SEM|n)\b[^)]*\)", "", line)
                        line = re.sub(r"\s+", " ", line).strip()
                        if line:
                            cleaned_lines.append(line)
                
                # Create new Document with cleaned content
                cleaned_content = "\n".join(cleaned_lines).strip()
                
                # Remove remaining statistical references
                cleaned_content = re.sub(r"\b(p|SD|SE|SEM)\b\s*[<>]?=\s*\d+\.?\d*", "", cleaned_content)
                cleaned_content = re.sub(r"\d+\s*±\s*\d+", "", cleaned_content)
                cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()
                
                if cleaned_content:
                    cleaned_doc = Document(page_content=cleaned_content, metadata=doc.metadata)
                    cleaned_documents.append(cleaned_doc)
                else:
                    logger.debug(f"Skipping empty document after cleaning (original metadata: {doc.metadata})")
                    cleaned_documents.append(doc)  # Returns original if empty after cleaning

            except Exception as e:
                logger.error(f"Error processing document with metadata {doc.metadata}: {str(e)}")
                cleaned_documents.append(doc)  # Returns original on error
                continue
        
        logger.info(f"Processed {len(documents)} documents, returned {len(cleaned_documents)} cleaned documents")
        return cleaned_documents

    clean_docs = preprocess_documents(pdf)
    return clean_docs

    # Not my code :(
    # Removes footers, page numbers, tables, and irrelevant statistical information from texts extracted from pdf
    import re
    from typing import List
    from langchain_core.documents import Document
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    def preprocess_documents(documents: List[Document]) -> List[Document]:
        """
        Preprocess LangChain Document objects to remove headers, footers, tables, and statistical data.
        
        Args:
            documents: List of LangChain Document objects, each containing page_content and metadata.
        
        Returns:
            List of modified Document objects with cleaned page_content, retaining original metadata.
        """
        # Patterns for headers and footers
        header_patterns = [
            r"^\s*(Appendix No\. \d+):.*",
            r"^\s*Table \d+.*",
            r"^\s*Figure \d+.*"
        ]
        footer_patterns = [
            r"Page \d+ of \d+",
            r"^\s*\d+\s*$"  # Standalone page numbers
        ]
        
        # Enhanced patterns for table and statistical data detection
        table_keywords = {
            "MEAN", "SD", "SEM", "N.S.", "p<", "p>", "p=", "t-test", "ANOVA",
            "±", "SE", "STD", "STDEV", "standard deviation", "standard error",
            "statistical", "significant", "significance", "TABLE", "RESULT",
            "GROUP", "CONTROL", "TREATMENT", "DOSE", "mg/kg", "mL/kg"
        }
        
        table_patterns = [
            r"^\s*(\d+\.\d+\s+){2,}",  # Multiple decimal numbers
            r"^\s*(\d+\s+){3,}",       # Multiple whole numbers
            r"^\s*\w+\s+\d+\s+\d+",    # Word followed by numbers (e.g., "Group 1 23 45")
            r"^\s*[A-Z][a-z]+\s+\d+",  # Capitalized word followed by numbers
            r"^\s*\d+\s*-\s*\d+",      # Number ranges
            r"^\s*\d+\s*/\s*\d+"       # Number ratios
        ]
        
        # Patterns for statistical results
        stat_patterns = [
            r"p\s*[<>]?\s*0\.\d+",     # p-values
            r"\d+\s*±\s*\d+",          # Mean ± SD
            r"\d+\s*\(\s*\d+\s*\)",    # Numbers in parentheses
            r"[A-Za-z]+\s*=\s*\d+",    # Statistical notation (e.g., "F = 23")
            r"r\s*=\s*-?\d+\.\d+"      # Correlation coefficients
        ]
        
        cleaned_documents = []
        
        for doc in documents:
            try:
                # Split page_content into lines
                lines = doc.page_content.split("\n")
                cleaned_lines = []
                in_table = False
                table_lines = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Skip headers
                    if any(re.match(pattern, line) for pattern in header_patterns):
                        logger.debug(f"Skipping header: {line}")
                        continue
                    
                    # Skip footers
                    if any(re.match(pattern, line) for pattern in footer_patterns):
                        logger.debug(f"Skipping footer: {line}")
                        continue
                    
                    # Skip lines that are entirely uppercase (often section headers or table headers)
                    if line.isupper() and len(line) > 3:
                        logger.debug(f"Skipping uppercase line (likely header): {line}")
                        continue
                    
                    # Detect table and statistical data
                    is_table_line = False
                    
                    # Check for table patterns
                    if any(re.match(pattern, line) for pattern in table_patterns):
                        is_table_line = True
                    
                    # Check for statistical keywords
                    if any(keyword.lower() in line.lower() for keyword in table_keywords):
                        is_table_line = True
                    
                    # Check for statistical patterns
                    if any(re.search(pattern, line) for pattern in stat_patterns):
                        is_table_line = True
                    
                    # Additional checks for tables
                    if line.count('\t') > 2 or line.count('  ') > 3:  # Multiple tabs or spaces
                        is_table_line = True
                    
                    if is_table_line:
                        in_table = True
                        table_lines += 1
                        logger.debug(f"Skipping table/statistical line: {line}")
                        continue
                    
                    # Reset table flag if we encounter a narrative line
                    if in_table and (len(line.split()) > 5 or len(line) > 60):
                        if table_lines < 3:  # Small tables might be false positives
                            cleaned_lines.extend(["[TABLE DATA REMOVED]"] * table_lines)
                        in_table = False
                        table_lines = 0
                    
                    # Keep non-table narrative text
                    if not in_table:
                        # Remove parenthetical statistical references
                        line = re.sub(r"\([^)]*\b(p|SD|SE|SEM|n)\b[^)]*\)", "", line)
                        line = re.sub(r"\s+", " ", line).strip()
                        if line:
                            cleaned_lines.append(line)
                
                # Create new Document with cleaned content
                cleaned_content = "\n".join(cleaned_lines).strip()
                
                # Remove remaining statistical references
                cleaned_content = re.sub(r"\b(p|SD|SE|SEM)\b\s*[<>]?=\s*\d+\.?\d*", "", cleaned_content)
                cleaned_content = re.sub(r"\d+\s*±\s*\d+", "", cleaned_content)
                cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()
                
                if cleaned_content:
                    cleaned_doc = Document(page_content=cleaned_content, metadata=doc.metadata)
                    cleaned_documents.append(cleaned_doc)
                else:
                    logger.debug(f"Skipping empty document after cleaning (original metadata: {doc.metadata})")
                    cleaned_documents.append(doc)  # Returns original if empty after cleaning

            except Exception as e:
                logger.error(f"Error processing document with metadata {doc.metadata}: {str(e)}")
                cleaned_documents.append(doc)  # Returns original on error
                continue
        
        logger.info(f"Processed {len(documents)} documents, returned {len(cleaned_documents)} cleaned documents")
        return cleaned_documents

    clean_docs = preprocess_documents(pdf)
    return clean_docs