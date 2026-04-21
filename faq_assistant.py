"""
AI-Powered FAQ Assistant using LangChain + Azure OpenAI.
Uses RAG (Retrieval Augmented Generation) to answer citizen questions
by retrieving relevant FAQ context before generating answers.
"""

import logging
from typing import Optional
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful government services assistant for citizens.
Answer questions clearly and accurately based on the provided FAQ context.
If you cannot find a confident answer in the context, say so and suggest contacting support.
Always be professional, empathetic, and concise.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""


class FAQAssistant:
    def __init__(self):
        self.settings = get_settings()
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()
        self.vector_store: Optional[FAISS] = None
        self.chain: Optional[ConversationalRetrievalChain] = None
        logger.info("FAQAssistant initialized")

    def _init_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=self.settings.azure_openai_endpoint,
            azure_deployment=self.settings.azure_openai_deployment_name,
            api_version=self.settings.azure_openai_api_version,
            api_key=self.settings.azure_openai_api_key,
            temperature=0.1,
            max_tokens=800,
        )

    def _init_embeddings(self) -> AzureOpenAIEmbeddings:
        return AzureOpenAIEmbeddings(
            azure_endpoint=self.settings.azure_openai_endpoint,
            azure_deployment=self.settings.azure_openai_embedding_deployment,
            api_version=self.settings.azure_openai_api_version,
            api_key=self.settings.azure_openai_api_key,
        )

    def load_knowledge_base(self, faq_documents: list[dict]) -> None:
        """Load FAQ documents into FAISS vector store."""
        docs = [
            Document(
                page_content=doc["content"],
                metadata={"category": doc.get("category", "general"), "id": doc.get("id")}
            )
            for doc in faq_documents
        ]
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        logger.info(f"Loaded {len(docs)} FAQ documents into vector store")
        self._build_chain()

    def _build_chain(self) -> None:
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=self.settings.conversation_memory_k,
            return_messages=True,
            output_key="answer"
        )
        prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "chat_history", "question"]
        )
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.max_retrieved_docs}
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False
        )

    def ask(self, question: str, session_id: str = "default") -> dict:
        """Process a citizen question and return AI-generated answer."""
        if not self.chain:
            raise RuntimeError("Knowledge base not loaded. Call load_knowledge_base() first.")

        logger.info(f"[{session_id}] Processing question: {question[:80]}...")
        result = self.chain.invoke({"question": question})

        sources = list({doc.metadata.get("category", "general") for doc in result.get("source_documents", [])})

        return {
            "answer": result["answer"],
            "sources": sources,
            "session_id": session_id,
        }
