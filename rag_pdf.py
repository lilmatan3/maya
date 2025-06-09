# rag_chat.py

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import os

# === יצירת מודל שיחה ===
model = OllamaLLM(model="aya")

# === תבנית שאלה-תשובה בעברית ===
template = """
המסמכים הרלוונטיים הם:
{reviews}

הנה השאלה:
{question}

ענה בצורה ברורה, בעברית, על בסיס המסמכים בלבד.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# === הגדרת בסיס הנתונים (Chroma) ===
db_location = "./chrome_langchain_db"
vector_store = Chroma(
    collection_name="hebrew_documents",
    persist_directory=db_location
)

# === רטריבר לחיפוש מסמכים ===
retriever = vector_store.as_retriever()

# === שיחה חיה ===
while True:
    print("\n" + "-"*40)
    question = input("✍️ שאל שאלה (q ליציאה): ")
    if question.strip().lower() == "q":
        break

    # שליפת מסמכים רלוונטיים
    reviews = retriever.invoke(question)

    # שליחה למודל
    result = chain.invoke({
        "reviews": reviews,
        "question": question
    })

    print("\n🤖 תשובה:", result)
