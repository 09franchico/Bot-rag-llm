from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI




set_debug(True)
set_verbose(True)


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "qwen2.5",online_llm = False):

        if online_llm:
            # self.model = ChatDeepSeek(
            #     model=llm_model,
            #     temperature=0,
            # )
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-001",
                temperature=0,
               )
        else:
            self.model = ChatOllama(model=llm_model)
       

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "Você é um assistente virtual prestativo e está respondendo perguntas gerais.Use os seguintes pedaços de contexto recuperado para responder à pergunta. Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa. Responda em português.",
                ),
                (
                    "human",
                    "Aqui estão as peças do documento: {context}\n pergunta: {question}",
                ),
            ]
        )

        self.vector_store = None
        self.retriever = None
        self.chain = None

    def ingest(self, pdf_file_path: str):
        
            
        #Ler documento
        #-----------------------------
        docs = PyPDFLoader(file_path=pdf_file_path).load()

        #Split nos documentos
        #-----------------------------
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        

        #Armazenamento e trasnformação dos embeddings
        #-----------------------------
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-de"),
            persist_directory="chroma_db",
        )


    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding=FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-de")
            )

        #Buscar dos dados do chromedb para adicionar no contexto -> Recuperador de texto as_retriever
        #-----------------------------
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.retriever.invoke(query)

        # Ajuste da corrente (chain) com a LLM
        #-----------------------------
        self.chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        if not self.chain:
            return "Por favor, adicione um documento PDF primeiro."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
