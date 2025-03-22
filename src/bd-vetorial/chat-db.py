import psycopg2
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

class ChatDB:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "qwen2.5", db_config=None):
        self.model = ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

        self.prompt = ChatPromptTemplate([
            (
                "system",
                "Você é um assistente virtual prestativo e responde perguntas com base no banco de dados. "
                "Se não souber a resposta, diga que não sabe. Mantenha a resposta curta e clara. Responda em português.",
            ),
            (
                "human",
                "Aqui estão as informações do banco de dados: {context}\nPergunta: {question}",
            ),
        ])

        self.db_config = db_config

    def fetch_data_from_db(self):
        """ Conecta ao PostgreSQL e busca os dados. """
        if not self.db_config:
            raise ValueError("Configuração do banco de dados não fornecida!")

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        # Ajuste essa consulta para trazer os dados relevantes
        cur.execute("SELECT id, descricao FROM tabela_dados;")
        rows = cur.fetchall()

        cur.close()
        conn.close()

        docs = [f"ID: {row[0]}, Descrição: {row[1]}" for row in rows]
        return docs

    def ingest(self):
        """ Busca dados do PostgreSQL e indexa no banco vetorial. """
        data = self.fetch_data_from_db()
        chunks = self.text_splitter.split_text("\n".join(data))
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory="chroma_db",
        )

    def ask(self, query: str):
        """ Faz uma busca no banco vetorial e responde com base nos dados. """
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding=FastEmbedEmbeddings()
            )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.retriever.invoke(query)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        if not self.chain:
            return "Por favor, adicione dados do banco de dados primeiro."

        return self.chain.invoke(query)
