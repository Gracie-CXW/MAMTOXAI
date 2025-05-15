# In Progress
# As of MAY-09-2025: Written some core tools for the AI to utilize. Still yet to implement chat memeory or tools to decode files other than pdfs. Currently, only one question can be asked, and no cache is available. 
# As of MAY-15-2025: Written tool for query breakdown, created pull request for using langchain built-in chroma module instead of chromadb itself in fear of compatibility issues. Updated parsing functions, original code was messy af.
# This file contains possible tools for the model to utilize

def empty_cache(fp):
    if os.path.exists(fp):
        shutil.rmtree(fp)
        os.makedirs(fp)

def store_pdf_id(pdf,dict):
    # This code looks trash, need to review later. Oringinally wanted to use for generating names for chromadb collections to avoid conflict, not sure how to implement.
    import numpy as np

    # generate and test id
    id = np.random.rand(1,1).tolist()[0][0]
    while dict[id]:
        id = np.random.rand(1,1).tolist()[0][0]
    
    # store
    dict.update({id:pdf})


def search_and_parse(pdf,keywords,vector_dir):
    import pymupdf
    import chromadb 
    import numpy as np

    doc = pymupdf.open(pdf)
    pages = [page.get_text() for page in doc] # room for opitmization, see if can use dicts, too lazy now
    relevant_pages = []

    # Find pages of the report relevant to the question, if the question is specific (e.g. about sensitization, genotox..etc.)
    for keyword in keywords:
        for p in pages:
            index = pages.index(p)
            text_lower = p.lower()
            keyword_lower = keyword.lower()
            words_page = text_lower.split(' ')
            title = " ".join(words_page[:100])

            if words_page.count(keyword_lower) > 0:
                relevant_pages.append([title,index,p])
            
    # Initializing vector client
    client = chromadb.PersistentClient(path=vector_dir)

    metadata = [{
        "original_doc_dir":pdf,
        "titles": title,
        "pages": index
        } for title,index,text in relevant_pages]
    
    ids = [f'page{i}' for i in range(len(relevant_pages))]

    texts = [page[2] for page in pages]
    collection_name = np.random.rand(1,1).tolist()[0][0]
    
    collection = client.create_collection(name=str(collection_name)) # Chroma uses SentenceTransformers as default, built-in embedding. Naming is random to avoid hitting existing names

    collection.add(
        documents = texts,
        metadatas = metadata,
        ids = ids
    )

def just_parse(pdf,vector_dir):
    import pymupdf 
    from sentence_transformers import SentenceTransformer
    import chromadb 

    doc = pymupdf.open(pdf)
    pages = [page.get_text() for page in doc]
    Title = ''.join(pages[0].split('\n')[:10])

    client = chromadb.PersistentClient(path=vector_dir)

    model = SentenceTransformer('all-MiniLM-L6-v2') 
    embeddings = model.encode(pages).tolist()

    collection = client.create_collection(name='toxicology_all',embedding_function=embeddings)
    collection.add(
        documents=pages,
        metadatas=[{
            'title':Title
        }]
    )

def breakdown_input(question,llm):
    from langchain import PromptTemplate
    llm=llm
    question=question
    template = """
    You are a toxicology evaluator at Health Canada. You are being asked a question on a toxicology report submission for a
    particular substance or chemical. Break down the question in terms that an LLM model can easily 
    understand and therefore answer. The question is as follows: {question}
    """
    prompt = PromptTemplate(
        input_variables=["query"],
        template=template
    )
    response = llm(prompt.format(query=question)

    return response.strip(\n) # not sure if correct syntax, should check later.

def RAG(question,vector_dir):
    question = question
    vector_dir = vector_dir
    

    class RAG_State(TypedDict):
        question: str
        context: 
