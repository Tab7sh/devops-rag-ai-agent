import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware

# 1. API Server Setup
app = FastAPI(title="Professional DevOps AI Assistant API")

# Setup CORS for frontend connection (shopandbid.com or localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. AI & Knowledge Base Setup (Wikipedia DevOps Page)
# !!! REPLACE WITH YOUR ACTUAL API KEY !!!
os.environ["OPENAI_API_KEY"] = "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"
os.environ["USER_AGENT"] = "Pro_DevOps_Assistant/1.0"
target_website = "https://en.wikipedia.org/wiki/DevOps" 

print(f"⏳ INITIALIZING SYSTEM: Loading knowledge from {target_website}...")

# Data Ingestion & Memory Setup
loader = WebBaseLoader(target_website)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(documents=chunks, embedding=embeddings)

# Professional Model Setup
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
retriever = db.as_retriever()

# Professional System Prompt (English)
system_prompt = (
    "You are an expert, professional DevOps Corporate Assistant. Answer the user's questions based strictly on the provided context."
    "Maintain a formal and helpful tone. If the information is not available in the context, state clearly: "
    "'I apologize, but I do not have specific information on that topic in my database.'"
    "\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("✅ SYSTEM READY: Professional DevOps Assistant is Online.")

# 3. API Endpoints
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        response = rag_chain.invoke({"input": request.message})
        return {"reply": response['answer']}
    except Exception as e:
        return {"reply": f"An unexpected error occurred: {str(e)}"}

@app.get("/")
def home():
    return {"status": "AI Agent is Online and Professional!"}
