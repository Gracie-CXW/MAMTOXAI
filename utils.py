# In Progress
# As of MAY-09-2025: Written some core tools for the AI to utilize. Still yet to implement chat memeory or tools to decode files other than pdfs. Currently, only one question can be asked, and no cache is available. 
# As of MAY-15-2025: Written tool for query breakdown, created pull request for using langchain built-in chroma module instead of chromadb itself in fear of compatibility issues. Updated parsing functions, original code was messy af.
# This file contains possible tools for the model to utilize

def empty_cache(fp):
    if os.path.exists(fp):
        shutil.rmtree(fp)
        os.makedirs(fp)

def search_and_parse(pdf: str, keywords: list[str], vector_dir: str):
    import pymupdf
    import chromadb
    from langchain_chroma import Chroma
    import uuid
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    doc = pymupdf.open(pdf)
    pages = [page.get_text() for page in doc] # room for opitmization, see if can use dicts, too lazy now
    relevant_pages = []

    # Find pages of the report relevant to the question, if the question is specific (e.g. about sensitization, genotox..etc.)
    try:
      for idx, p in enumerate(pages):
          index = idx
          text_lower = p.lower()
          words_page = text_lower.split(' ')
          title = " ".join(words_page[:100])

          if any(keyword.lower() in text_lower for keyword in keywords):
              relevant_pages.append([title, index, p])

      documents = [Document(
          page_content = text,
          metadata = {"original_doc_dir":pdf,
          "titles": title,
          "page": index},
          id = index
      ) for title,index,text in relevant_pages]

      uuids = [str(uuid.uuid4()) for _ in range(len(documents))]

      collection_name = str(uuid.uuid4())  # Unique collection name

      # Initialize Vector Store
      embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
      vector_store = Chroma(
          collection_name=collection_name,
          embedding_function=embedding_func,
          persist_directory=vector_dir
      )

      # Add pages to VDB
      if relevant_pages:  # Only add if there are relevant pages
          vector_store.add_documents(documents=documents,ids=uuids)

      # For one pdf only
      return vector_store

      # For multiple PDFs, simply create global dictionary and update.
    except FileNotFoundError:
        raise ValueError(f"PDF file not found: {pdf}")
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def just_parse(pdf,vector_dir):
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
      embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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

def calculator_add(*args):
    return sum(args)

def calculator_multiply(*args):
    from math import prod
    return prod(args) if args else 1

