import os

if os.getenv("VECTOR_DB") == "PINECONE":

    import os
    import hashlib
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    from pathlib import Path
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer

    # Set up Pinecone
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX = os.getenv('COLLECTION_NAME')
    print("initialising Pinecone connection...")
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY, source_tag="cloudera")
    print("Pinecone initialised")

    if PINECONE_INDEX not in pinecone_client.list_indexes().names():
        print(f"Creating 768-dimensional index called '{PINECONE_INDEX}'...")
        pinecone_client.create_index(name=PINECONE_INDEX, dimension=768)
        print("Index creation successful")
    else:
        print(f"Index '{PINECONE_INDEX}' already exists")

    pinecone_index = pinecone_client.Index(PINECONE_INDEX)

    # Tokenizer and Model
    EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_REPO)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_REPO)

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(sentence):
        encoded_input = tokenizer(sentence, padding='max_length', truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()[0]

    def insert_embedding(pinecone_index, id_path, text):
        print(f"Upserting vectors for file path: {id_path}")
        unique_id = hashlib.sha256(id_path.encode()).hexdigest()[:10]  # Shorten the hash to 10 characters
        vectors = list(zip([unique_id], [get_embeddings(text)], [{"file_path": id_path}]))
        upsert_response = pinecone_index.upsert(vectors=vectors)
        print("Upsert successful")

    def main():
        doc_dir = '/home/cdsw/data'
        for file in Path(doc_dir).glob('**/*.txt'):
            with open(file, "r") as f:
                text = f.read()
                print(f"Generating embeddings for: {file.name}")
                insert_embedding(pinecone_index, os.path.abspath(file), text)
        print('Finished loading Knowledge Base embeddings into Pinecone')

    if __name__ == "__main__":
        main()



if os.getenv("VECTOR_DB") == "CHROMA":

    ## Initialize a connection to the running Chroma DB server
    import chromadb
    from pathlib import Path

    # chroma_client = chromadb.Client()
    chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma-data")

    from chromadb.utils import embedding_functions
    EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
    EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    COLLECTION_NAME = os.getenv('COLLECTION_NAME')

    print("initialising Chroma DB connection...")

    print(f"Getting '{COLLECTION_NAME}' as object...")
    try:
        chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
        print("Success")
        collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    except:
        print("Creating new collection...")
        collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
        print("Success")

    # Get latest statistics from index
    current_collection_stats = collection.count()
    print('Total number of embeddings in Chroma DB index is ' + str(current_collection_stats))

    # Helper function for adding documents to the Chroma DB
    def upsert_document(collection, document, metadata=None, classification="public", file_path=None):
        
        # Push document to Chroma vector db (if file path is not available, will use first 50 characters of document)
        if file_path is not None:
            response = collection.add(
                documents=[document],
                metadatas=[{"classification": classification}],
                ids=[file_path]
            )
        else:
            response = collection.add(
                documents=[document],
                metadatas=[{"classification": classification}],
                ids=document[:50]
            )
        return response

    # Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
    def load_context_chunk_from_data(id_path):
        with open(id_path, "r") as f: # Open file in read mode
            return f.read()

    # Read KB documents in ./data directory and insert embeddings into Vector DB for each doc
    doc_dir = '/home/cdsw/data'
    for file in Path(doc_dir).glob(f'**/*.txt'):
        print(file)
        with open(file, "r") as f: # Open file in read mode
            print("Generating embeddings for: %s" % file.name)
            text = f.read()
            upsert_document(collection=collection, document=text, file_path=os.path.abspath(file))
    print('Finished loading Knowledge Base embeddings into Chroma DB')
