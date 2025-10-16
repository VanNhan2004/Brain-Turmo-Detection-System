# main.py
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# -----------------------------
# Cấu hình thư mục
PDF_FOLDER = "./chatbot/file"
VECTOR_DB_PATH = "./chatbot/vector_db/faiss_index"
# -----------------------------

# 1. Load PDF và tách thành chunks
def load_pdfs(folder_path=PDF_FOLDER, chunk_size=1000, chunk_overlap=200):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", ".", "?", "!", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)
    return all_docs

# 2. Tạo hoặc load FAISS vectorstore
def create_vectorstore(documents, save=True):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    
    if save:
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        vectorstore.save_local(VECTOR_DB_PATH)
        print(f"Vectorstore đã được lưu tại {VECTOR_DB_PATH}")
    return vectorstore

def load_vectorstore():
    if not os.path.exists(VECTOR_DB_PATH) or not os.listdir(VECTOR_DB_PATH):
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    print(f"Đã load vectorstore từ {VECTOR_DB_PATH}")
    return vectorstore

# 3. Tạo hybrid retriever (FAISS + BM25)
def get_hybrid_retriever(vectorstore, faiss_k=3, bm25_k=3):
    retriever_faiss = vectorstore.as_retriever(search_kwargs={"k": faiss_k})
    top_docs = vectorstore.similarity_search("", k=100)
    retriever_bm25 = BM25Retriever.from_documents(top_docs)
    retriever_bm25.k = bm25_k

    return EnsembleRetriever(
        retrievers=[retriever_faiss, retriever_bm25],
        weights=[0.7, 0.3]
    )

# 4. Khởi tạo prompt chatbot
def get_chat_prompt():
    return ChatPromptTemplate.from_template("""
Bạn là chatbot AI tên là ChatBot Doctor, trả lời các câu hỏi dựa trên tài liệu được cung cấp, trong lĩnh vực các bệnh về u não .

Yêu cầu:
- Trả lời ngắn gọn, rõ ràng, dễ hiểu.
- Chỉ sử dụng dữ liệu trong phần CONTEXT, tuyệt đối không tự thêm thông tin khác.
- Nếu CONTEXT có nhiều chi tiết liên quan, hãy liệt kê đầy đủ và mạch lạc.
- Không bao giờ được nhắc đến từ CONTEXT trong câu trả lời của bạn.
- Không bịa thông tin, không suy đoán.
- Nếu không có dữ liệu, trả lời chính xác:
  "Xin lỗi, tôi không có dữ liệu về vấn đề này."
- Đảm bảo rằng tất cả thông tin bạn đưa ra đều có trong CONTEXT.
- Khi người dùng sử dụng những lời hỏi thăm chào hỏi hoặc cảm xúc thì bạn nên trả lời họ và hỏi họ có cần giúp gì không.                            

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:
""")

# 5. Khởi tạo LLM Llama 3.2
def init_llm(model_name="llama3.2"):
    return ChatOllama(model=model_name)

# 6. Hỏi câu hỏi chatbot
def ask_question(retriever, question, llm, prompt):
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    prompt_text = prompt.format(context=context, question=question)
    result = llm.invoke(prompt_text)
    return result.content

# -----------------------------
# MAIN
# -----------------------------
def main():
    # Load hoặc tạo vectorstore
    vector_store = load_vectorstore()
    if vector_store is None:
        print("Tạo FAISS vectorstore mới từ PDF...")
        documents = load_pdfs()
        vector_store = create_vectorstore(documents)
    
    # Khởi tạo retriever, prompt, LLM
    retriever = get_hybrid_retriever(vector_store)
    prompt = get_chat_prompt()
    llm = init_llm()

    print("\n=== ChatBot đã sẵn sàng ===")
    print("Gõ 'exit' để thoát.\n")

    while True:
        user_question = input("User: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        answer = ask_question(retriever, user_question, llm, prompt)
        print("Chatbot:", answer, "\n")

if __name__ == "__main__":
    main()

# Thêm vào cuối chatbot.py

def init_chatbot():
    vector_store = load_vectorstore()
    if vector_store is None:
        documents = load_pdfs()
        vector_store = create_vectorstore(documents)
    
    retriever = get_hybrid_retriever(vector_store)
    prompt = get_chat_prompt()
    llm = init_llm()
    return retriever, prompt, llm

def ask_chat(retriever, prompt, llm, question):
    return ask_question(retriever, question, llm, prompt)
