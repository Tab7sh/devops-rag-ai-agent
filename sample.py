import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 0. SETUP: Apni API Key aur Website ka link dalein
# ==========================================
os.environ["OPENAI_API_KEY"] = "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"
target_website = "https://shopandbid.com" # Isko change kar ke kisi bari website ka link dalein

print("🚀 AI Agent Start ho raha hai...")

# ==========================================
# 1. DATA INGESTION (Website Parhna)
# ==========================================
print(f"🌐 {target_website} ka data parh raha hoon...")
loader = WebBaseLoader(target_website)
data = loader.load()

# ==========================================
# 2. CHUNKING (Tukray Karna)
# ==========================================
print("✂️ Text ko tukron mein tor raha hoon...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(data)

# ==========================================
# 3. VECTOR DATABASE (Memory Banana)
# ==========================================
print("🧠 Data ko memory (ChromaDB) mein save kar raha hoon. Thora wait karein...")
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)
print("✅ Memory setup complete!\n")

# ==========================================
# 4. AI AGENT SETUP (Brain & Logic)
# ==========================================
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
retriever = db.as_retriever()

system_prompt = (
    "You are a professional assistant. User ke sawal ka jawab sirf neechay diye gaye Context ko parh kar dein. "
    "Agar jawab context mein nahi hai, toh safai se keh dein ke 'Mujhe nahi maloom'. Apni taraf se jhoot mat bolein."
    "\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ==========================================
# 5. CHATBOT LOOP (Live Baat-Cheet)
# ==========================================
print("="*60)
print("💬 Chat Shuru! (Agent se baat karein. Khatam karne ke liye 'exit' type karein)")
print("="*60)

while True:
    user_question = input("\n👤 Aap: ")
    
    if user_question.lower() == 'exit':
        print("🤖 AI Agent: Allah Hafiz! Chat band ho rahi hai.")
        break
        
    if user_question.strip() != "":
        print("🤖 AI soch raha hai...")
        try:
            # AI se jawab nikalwana
            response = rag_chain.invoke({"input": user_question})
            print(f"\n🤖 AI Agent: {response['answer']}")
        except Exception as e:
            print(f"\n❌ Ek masla aagaya: {e}")
        
        print("-" * 60)