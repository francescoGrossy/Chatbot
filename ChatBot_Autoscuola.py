from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import create_db

# Carica API Key da .env
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Rispondi solo basandoti sulle seguenti informazioni:

{context}

---

Domanda dell'utente: {question}
"""

def query_chroma(query_text, db):
    """Interroga il database Chroma con la domanda dell'utente."""
    # Cerca la domanda più simile nel database
    results = db.similarity_search_with_relevance_scores(query_text, k=2)

    if not results or results[0][1] < 0.7:
        print("Mi dispiace, non ho trovato una risposta nel materiale dell'autoscuola.")
        return

    # Costruisce il contesto basato sulle risposte trovate
    context_text = "\n\n".join([doc.page_content for doc, _ in results])

    # Prepara il prompt per OpenAI
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Modello OpenAI
    model = ChatOpenAI()
    response_text = model.invoke(prompt).content

    print(f"Risposta: {response_text}\n")

def main():
    # Crea il database solo se non esiste
    if not os.path.exists(CHROMA_PATH):
        print("📂 Creazione del database Chroma...")
        create_db.main()

    # Carica il database Chroma
    print("🔍 Caricamento del database Chroma completato.")
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Avvia il chatbot in un loop
    print("\n🤖 Chatbot avviato! Digita 'exit' per uscire.")
    while True:
        query_text = input("📢 Fai una domanda sull'autoscuola: ")
        if query_text.lower() == "exit":
            print("👋 Arrivederci!")
            break
        query_chroma(query_text, db)

if __name__ == "__main__":
    main()
