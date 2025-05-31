# In Progress
# As of MAY-09-2025: Written some core tools for the AI to utilize. Still yet to implement chat memeory or tools to decode files other than pdfs. Currently, only one question can be asked, and no cache is available. 
# As of MAY-15-2025: Written tool for query breakdown, created pull request for using langchain built-in chroma module instead of chromadb itself in fear of compatibility issues. Updated parsing functions, original code was messy af.
# This file contains possible tools for the model to utilize

# Last Updated: May-29, 3:03am
import sys

def empty_cache(fp):
    if os.path.exists(fp):
        shutil.rmtree(fp)
        os.makedirs(fp)

def save_vector_store(persist_directory:str,collection_name:str,original_doc:str,ids:str,vector_save_fp:str):
    """
    Input Arguments:
    Persist_directory: name (filepath) of the directory storing the embedding. str
    Original Document: name of the original document, not the filepath. str
    Collection_name: unique name (uuid) of the embedding collection
    Vector Save Directory Filepath: str. 

    Output:
    None
    
    Save file has following syntax: 
    persist_directory (key) -> str : [original_doc,collection_name,ids] (value) -> List(str)
    """
    import json
    try:
        with open(vector_save_fp, 'a') as save:
            save.write(json.dumps({persist_directory: [original_doc, collection_name,ids]}) + '\n')

            print(f'successfully saved vector store for document {original_doc} to file {vector_save_fp}.')
    except Exception as e:
        raise Exception(f"Error saving vector store for document {original_doc} to file {vector_save_fp}: {str(e)}")

def load_dict_from_txt(save_fp):
    import json
    import os
    """
    Input argument:
    save_fp: str, filepath of the saved dictionary.

    Output:
    Dictionary Object.
    """

    result = {}

    if not os.path.exists(save_fp):
        print(f"File not found: {save_fp}")
        return result

    with open(save_fp, 'r') as f:
        lines = f.readlines()
        if not lines:
            print(f"Cannot load dict from file {save_fp}; File is empty.")
            return result

        for line in lines:
            try:
                entry = json.loads(line.strip())
                if isinstance(entry, dict):
                    result.update(entry)
                else:
                    print(f"Warning: Line is not a valid dict: {line.strip()}")
            except json.JSONDecodeError as e:
                print(f"Failed to parse line: {line.strip()} - {e}")

    return result

def just_parse(pdf:str,vector_dir:str):
    """
    Input Arguments: 
    PDF file path: str 
    Vector Database Directory: str

    Outputs:
    vector_store: langchain_chroma.Chroma Object -> The embedded values
    persist_directory: str -> Directory of stored embedding.
    collection_name: str -> collection name.
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
      text = "\n\n".join(pages)
      uuid_doc = str(uuid.uuid4())
      uuids_pages = []
      for _ in range(len(pages)):
        uuids_pages.append(str(uuid.uuid4()))
      documents = [Document(
          page_content = text,
          metadata = {'title':Title,
          'length':len(pages),
          'parse function':'just_parse'},
          id = uuid_doc
      )]

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
              ids = uuids_pages)

      # For multiple pdfs
      
      return vector_store,persist_directory,collection_name

    except FileNotFoundError:
        raise ValueError(f"PDF file not found: {pdf}")
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def calculator_add(*args):
    # Specifically for calculating gross sample sizes
    return sum(args)

def calculator(operation,*args):
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

def search_and_parse(pdf: str, keywords: list[str], vector_dir: str):
    """
    Input Arguments:
    PDF file path: str
    Keywords: list of str, only matches pages with ALL keywords present. 
    Vector Database Directory path: str

    Outputs:
    vector_store: langchain_chroma.Chroma Object -> Only pages containing all keywords
    persist_directory: str -> Directory of stored embedding. In format '(pdf name)_(keywords)_searchandparse'
    collection_name: str -> collection name 
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
    keywords_name = "_".join(keywords)
    modified_name = f"{pdf_name}_{keywords_name}_searchandparse"

    doc = pymupdf.open(pdf)
    pages = [page.get_text() for page in doc]
    relevant_pages = []
    toc = doc.get_toc()
    keywords_lower = [k.lower() for k in keywords]
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
              section_pages = list(range(start_page, end_page + 1))
              section_texts = [pages[s] for s in section_pages]

              combined_text = "\n".join(section_texts)

              # Include all pages in section if any page contains a keyword
              if all(k in combined_text for k in keywords_lower):
                section_title = sections[i][1]
                for p in section_pages:
                    relevant_pages.append([section_title, p, pages[p]])
      else:
          # Fallback to original page-by-page approach
          for idx, p in enumerate(pages):
              text_lower = p.lower()
              if any(keyword.lower() in text_lower for keyword in keywords):
                  title = " ".join(p.split()[:100])
                  relevant_pages.append([title, idx, p])

      # Create Document objects
      documents = [Document(
          page_content=text,
          metadata={"original_doc_dir": pdf, "titles": title, "page": index,'parse function':'search_and_parse'},
          id=str(index)
      ) for title, index, text in relevant_pages]

      ids = [str(index) for title,index,text in relevant_pages]
      collection_name = str(uuid.uuid4())  # Unique collection name

    # Initialize Vector Store
    # need to download embedding function bc for some proxy error for some reason.
      embedding_func = HuggingFaceEmbeddings(model_name=r".\embeddings_local\all-MiniLM-L6-v2")
      persist_directory = os.path.join(vector_dir,modified_name)
      vector_store = Chroma(
          collection_name=collection_name,
          embedding_function=embedding_func,
          persist_directory=persist_directory # Creates folder to store vector store
      )

      # Add pages to vector store
      if relevant_pages:
          vector_store.add_documents(documents=documents, ids=ids)

    except FileNotFoundError:
      raise ValueError(f"PDF file not found: {pdf}")
    except Exception as e:
      raise Exception(f"Error processing PDF: {str(e)}")

    return vector_store,persist_directory,collection_name

def max_min(operation: str,*args):
    try:
        if operation == 'max':
            return max(*args)
        elif operation == 'min':
            return min(*args)
    except Exception as e:
        print (f'Error while performing max_min calculation: {str(e)}')

def cleanup_doc(pdf):
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