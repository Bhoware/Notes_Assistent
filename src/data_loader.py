from pathlib import Path
from typing import List,Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader

def load_docs(data_dir:str)->list[Any]:

    data_path=Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_path}")

    docs=[]

    pdf_file=list(data_path.glob("*.pdf"))

    print(f"Found {len(pdf_file)} pdf files ")

    for file in pdf_file:
        try:
            loader = PyPDFLoader(str(file))
            loaded_pages = loader.load()
            docs.extend(loaded_pages)
            print(f"Loaded {len(loaded_pages)} pages from {file.name}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return docs
    
    
    

