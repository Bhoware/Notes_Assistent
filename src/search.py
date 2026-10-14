import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src.vectorstore import FaissVectorStore

load_dotenv()

class Search:
    def __init__(self, persist_dir: str = "faiss_store", model_name: str = "all-MiniLM-L6-v2", llm: str = "openai/gpt-oss-120b"):
        self.store = FaissVectorStore(persist_dir=persist_dir, embedding_model=model_name)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not os.path.exists(faiss_path) and os.path.exists(meta_path):
            from data_loader import load_docs
            docs=load_docs("./data/Documents")
            self.store.build_from_documents(docs)
        else:    
          self.store.load()
        self.llm=ChatGroq(model=llm,temperature=0)

    def search(self,query:str,k:int=5):
        result=self.store.query(query,k)
        context_list=[doc['metadata']['text'] for doc in result]
        context="\n\n".join(context_list)
        if not context:
            return "No relevant information found"   
        prompt = PromptTemplate(
                                input_variables=["context", "query"],
                                template="""
                                Answer the question using ONLY the context below. Cite the source for each claim.

                                Context:
                                {context}

                                Question: {query}

                                Answer:
                                """)
        chain=prompt | self.llm
        response=chain.invoke({"context":context,"query":query})
        return response.content 

        