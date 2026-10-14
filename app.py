from fastapi import FastAPI
from pydantic import BaseModel

from src.search import Search
from src.vectorstore import FaissVectorStore

app=FastAPI()

store=FaissVectorStore()
store.load()
search=Search()

class Query(BaseModel):
    query:str
    k:int=5

@app.post("/query")
def ask_ai(query:Query):
    return search.search(query.query,query.k)
    