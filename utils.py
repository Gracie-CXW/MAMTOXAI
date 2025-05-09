# In Progress
# As of MAY-09-2025: Written some core tools for the AI to utilize. Still yet to implement chat memeory or tools to decode files other than pdfs. Currently, only one question can be asked, and no cache is available. 

# This file contains possible tools for the model to utilize

def empty_cache(fp):
    if os.path.exists(fp):
        shutil.rmtree(fp)
        os.makedirs(fp)

def search_and_parse(pdf,keywords,vector_dir):
    import pymupdf
    import Sentence_Transformer
    import chromadb 
    from chromadb.configs import settings 

    doc = pymupdf.open(pdf)
    relevant_pages = []

    # Find pages of the report relevant to the question, if the question is specific (e.g. about sensitization, genotox..etc.)
    for keyword in keywords:
        for page in doc:
            if page.search_for(keyword) is not None:

                this_page = page.get_text()
                relevant_pages.append(this_page)

    # Initializing vector client
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",persist_directory=vector_dir))
    collection = client.get_or_create_collection(name="MAMTOX_Collections") 

    model = Sentence_Transformer('all-MiniLM-L6-v2') 
    current_title = None 
    current_txt = ""
    clean_pages = []

    for page in relevant_pages:
        if not page:
            continue 
        else:
            current_title = page.split('\n')[:15]
            current_text = page

            clean_pages.append([current_title,current_text])
    
    texts = [t for _,t in clean_pages]
    metatdata = [{"title":title} for title,_ in clean_pages]
    ids = [f'page{i}' for i in range(clean_pages)]

    embeddings = model.encode(texts).to_list()

    collection.add(
        documents = texts,
        metadata = metadata,
        ids = ids,
        embeddings = embeddings
    )

    client.persist()

def just_parse(pdf,vector_dir):
    import pymupdf 
    import Sentence_Transformer
    import chromadb 
    from chromadb.configs import settings 

    doc = pymupdf.open(pdf)
    pages = [page.get_text() for page in doc]

    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",persist_directory=vector_dir))
    collection = client.get_or_create_collection(name="MAMTOX_Collections")

    model = Sentence_Transformer('all-MiniLM-L6-v2') 
    embeddings = model.encode(texts).to_list()

    collection.add(
        documents = texts,
        embeddings = embeddings
    )

    client.persist()

