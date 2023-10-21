import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template

# Langchain Core module
#1 Agent with standard tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMMathChain
from langchain.utilities import SerpAPIWrapper
from langchain.tools import BaseTool, StructuredTool, Tool, tool

#1 model I/O
from langchain.chat_models import ChatOpenAI
#2 Data Connectivity
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
#https://instructor-embedding.github.io/ vs. openai/pricing
from langchain.vectorstores import FAISS #able to run locally
#3 Chain
from langchain.chains import ConversationalRetrievalChain #chain
from langchain.chains import RetrievalQA
#4 Memory
from langchain.memory import ConversationBufferMemory
#5 other components
from langchain.llms import HuggingFaceHub 
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:") #error inside main()

def get_pdf_text(pdf_docs):
    text = "" #init
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
#    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain_memory_llm_retriever_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory( # init an instance of memory, other kinds(entity memory)
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm( #able to chat with our context
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    retriever_chain= RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever())
    return conversation_chain, memory, llm, retriever_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question}) #st.session_state remembers every config
    st.session_state.chat_history = response['chat_history']
    search = SerpAPIWrapper()
#    llm_math_chain = LLMMathChain(llm=st.session_state.llm, verbose=True)
    tools = [
        Tool.from_function(
        name = "Document Store",
        func = st.session_state.llm_papers.run,
        description = "Use it to lookup information from the document \
                        store"
    ),
        Tool.from_function(
            func=search.run,
            name="Search",
            description="useful for when you need to answer questions about current events"
            # coroutine= ... <- you can specify an async method if desired as well
        ),
    ]

    conversational_agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    #agent='chat-conversational-react-description',
    tools=tools,
    llm=st.session_state.llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=st.session_state.memory,
    )

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    st.write(conversational_agent.run(user_question))    
    

def main():
    load_dotenv()
    
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:#before any conversation
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:#before any chat
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
#                st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
#                st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain_memory_llm_retriever_chain( #generate new conversation, save st from reloading every variable by making it static
                    vectorstore)[0]
                
                # memory
                st.session_state.memory = get_conversation_chain_memory_llm_retriever_chain(vectorstore)[1]
                #llm
                st.session_state.llm = get_conversation_chain_memory_llm_retriever_chain(vectorstore)[2]
                #llm papers(retriever chain)
                st.session_state.llm_papers = get_conversation_chain_memory_llm_retriever_chain(vectorstore)[3]

#   able to use st.session_state.conversation
    #st.session_state.conversation>>line 76~79 


if __name__ == '__main__':
    
    main()
