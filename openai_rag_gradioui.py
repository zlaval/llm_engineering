from functools import partial

import gradio as gr
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

load_dotenv()
openai = OpenAI()

utf8_loader = partial(TextLoader, encoding="utf-8")
loader = DirectoryLoader("rag", glob="**/*.txt", loader_cls=utf8_loader)
# TextLoader("rag/sample1.txt", encoding="utf-8"))
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
# Faiss is a library for efficient similarity search and clustering of dense vectors.
# It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
# It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy.
# Some of the most useful algorithms are implemented on the GPU.
# It is developed primarily at Meta's Fundamental AI Research group.
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) #only most relevant doc

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# It always sends context
#it has memory and chat history
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    callbacks=[StdOutCallbackHandler()],
    return_source_documents=True
)


def chat(prompt,history):
    result = chain.invoke({"question": prompt})
    print("Source doc:", [doc.metadata.get("source", "") for doc in result["source_documents"]])
    print("Memory:", memory.chat_memory.messages)
    return result["answer"]


gr.ChatInterface(fn=chat).launch()
