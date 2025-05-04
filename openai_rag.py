from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

load_dotenv()
openai = OpenAI()

loader = TextLoader("assets/sample.txt", encoding="utf-8")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

while True:
    question = input("🟢 Kérdés: ")
    if question.lower() in ["exit", "quit"]:
        break
    result = chain.invoke({"question": question})

    print("\n🟡 Válasz:", result["answer"])
    print("📄 Forrásdokumentum(ok):", [doc.metadata.get("source", "") for doc in result["source_documents"]])
    print("🧠 Memória állapot:", memory.chat_memory.messages)
