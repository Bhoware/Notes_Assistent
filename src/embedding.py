from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.data_loader import load_docs


class Embeddings:
    def __init__(self,model_name:str="sentence-transformers/all-MiniLM-L6-v2",chunk_size:int =1000,chunk_overlap:int=200):
        self.model = SentenceTransformer(model_name)
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        print(f"[INFO] Loaded embedding model:{model_name}")
        print(f"[INFO] Chunk size:{chunk_size}, Chunk overlap:{chunk_overlap}")
        
    def chunk_docs(self,docs: list[Document]) -> list[Document]:
        """
        Split documents into chunks
        """    

        splitters=RecursiveCharacterTextSplitter(
            separators=[" ","\n",], 
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        chunks=splitters.split_documents(docs)    
        
        print(f"[INFO] Split {len(docs)} documents into {len(chunks)} chunks")    
        return chunks
        
    def embed_texts(self,texts:list[str])->list[list[float]]:
        """
        Generate embeddings for texts
        """
        print(f"[INFO] Generating embeddings for {len(texts)} texts")
        embeddings=self.model.encode(texts,batch_size=32,show_progress_bar=True,convert_to_numpy=True)
        print("[INFO] Generated embeddings")
        return embeddings.tolist()    

    def embed_docs(self, docs: list[Document]) -> list[list[float]]:
        """
        Generate embeddings for documents
        """
        print(f"[INFO] Generating embeddings for {len(docs)} documents")
        texts = [doc.page_content for doc in docs]
        return self.embed_texts(texts)    
        
    def process_data(self,docs_path:str)->dict:
        """
        Process documents by chunking and embedding
        """
        print(f"[INFO] Processing documents from {docs_path}")
        docs=load_docs(docs_path)
        chunks=self.chunk_docs(docs)
        embeddings=self.embed_docs(chunks)
        return {"chunks":chunks,"embeddings":embeddings}