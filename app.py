from src.data_loader import load_docs
from src.embedding import Embeddings
from src.vectorstore import FaissVectorStore
from src.search import Search

if __name__=='__main__':
    store=FaissVectorStore()
    store.load()
    search=Search()
    print(search.search("Explain Backpropagation in detail",3))
    