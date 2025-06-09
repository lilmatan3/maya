from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# יצירת המודל
model = OllamaLLM(model="aya")

# תבנית Prompt פשוטה עם משתנה
template = """אתה עוזר חכם. אנא ענה על השאלה הבאה:
שאלה: {question}
תשובה:"""

# יצירת ה־Prompt Template
prompt = ChatPromptTemplate.from_template(template)

# חיבור ה־Prompt עם המודל
chain = prompt | model

# דוגמה להרצה
response = chain.invoke({"question": "מהי מדינת חוק?"})
print(response)
