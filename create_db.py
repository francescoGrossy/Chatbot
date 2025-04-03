import json
import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

CHROMA_PATH = "chroma"
DATA_PATH = "data\\books\\faq_autoscuola.json"

def load_json_data():
    """Carica le domande e risposte dal file JSON."""
    with open(DATA_PATH, "r", encoding="latin1") as file:
        data = json.load(file)
    return data

def create_documents(data):
    """Converte il JSON in oggetti Document per ChromaDB."""
    documents = []
    for item in data:
        doc = Document(
            page_content=f"Domanda: {item['question']}\nRisposta: {item['answer']}",
            metadata={"source": item.get("source", "Sconosciuto")}
        )
        documents.append(doc)
    return documents

def save_to_chroma(documents):
    """Crea il database Chroma con le domande dell'autoscuola."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Cancella il vecchio database

    db = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )
    print(f"Salvate {len(documents)} domande nel database.")

def main():
    data = load_json_data()
    documents = create_documents(data)
    save_to_chroma(documents)

if __name__ == "__main__":
    main()
