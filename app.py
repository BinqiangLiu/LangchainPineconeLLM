import streamlit as st
from streamlit_chat import message
import pinecone
import os
import openai
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, LLMChain
from pathlib import Path
from time import sleep
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="LangChain+Pincecone+LLM", page_icon=":robot:")

css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

st.header("AI Doc-Chatbot Demo")
st.write("For current App version: Docs are pre-embedded and stored on Pinecone.")

#OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
#pc_environment=os.environ["PINECONE_ENV"]
#pc_api_key=os.environ["PINECONE_API_KEY"]
#pc_index=os.environ["PINECONE_INDEX_NAME"]

#OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
#pc_environment=os.getenv("PINECONE_ENV")
#pc_api_key=os.getenv("PINECONE_API_KEY")
#pc_index=os.getenv("PINECONE_INDEX_NAME")

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
pc_environment=os.environ.get("PINECONE_ENV")
pc_api_key=os.environ.get("PINECONE_API_KEY")
pc_index=os.environ.get("PINECONE_INDEX_NAME")

embeddings = OpenAIEmbeddings()

# create a pinecone index
#pinecone.create_index("index name here", dimension=1536, metric="cosine") # 1536 dim of text-embedding-ada-002

# initialize pinecone
pinecone.init(
        api_key=pc_api_key,  # find api key in console at app.pinecone.io
        environment=pc_environment  # find next to api key in console
)

# create a loader
#loader = PyPDFLoader("valuation.pdf")

#The following two loaders can also be used for other PDF files:
#loader = UnstructuredPDFLoader("../data/summary_strategy.pdf")
#loader = OnlinePDFLoader("...")

# load your data
#data = loader.load()

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#texts = text_splitter.split_documents(data)

#docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=pc_index)
#docsearch = Pinecone.from_texts(  [t.page_content for t in texts], embeddings,  index_name=index_name, namespace=namespace)

#If you already have an index, you can load it like this
docsearch = Pinecone.from_existing_index(pc_index, embeddings)
#docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

#query = "NEO Chair"
#docs = docsearch.similarity_search(query) #默认返回4项结果？
#docs = docsearch.similarity_search(query,  include_metadata=True, namespace=namespace)
#print(docs)
#st.write(docs)

#print("Stop for Pinecone")
#st.write("Stop for Pinecone")
#st.stop() #先创建Pinecone向量数据库
#print("Stop for Pinecone")
#print("***********************")
#st.write("Stop for Pinecone")
#st.write("---")

def load_chain():
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')
    chain = load_qa_chain(llm, chain_type="stuff")
#    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

#chain = load_qa_chain(llm=llm, chain_type="stuff")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("Enter your question here: ", "Hi.", key="input")
#    input_text = st.text_input("You: ", "Hello, how are you?", key="input")  
    return input_text

user_input = get_text()
output=""
if user_input:
#   output = chain.run(input=user_input)
    sms_docs=docsearch.similarity_search(user_input)
    output = chain.run(input_documents=sms_docs, question=user_input)
  
#if user_input:
#   docs = chain.similarity_search(user_input)
#   output = docs[0].page_content

st.session_state.past.append(user_input)
st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


