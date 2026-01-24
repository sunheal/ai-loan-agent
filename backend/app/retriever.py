from typing import List
import os

from backend.app.utils import get_logger

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logger = get_logger("retriever")

class LoanKnowledgeRetriever:
  """
  Handles:
  - Loading loan documents
  - Chunking
  - Embedding
  - Vector search
  """

  def __init__(
    self,
    docs_path: str = "/app/docs",
    chunk_size: int = 500,
    chunk_overlap: int = 100
  ):
    self.docs_path = docs_path
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

    self.embeding_model = OpenAIEmbeddings()
    self.vectorstore = None


  # -------------------------
  # 1. Load Documents
  # -------------------------
  def load_documents(self) -> List[Document]:
    documents = []

    for filename in os.listdir(self.docs_path):
      if not filename.endswith((".txt", ".md")):
        continue

      file_path = os.path.join(self.docs_path, filename)
      loader = TextLoader(file_path, encoding = "utf-8")
      documents.extend(loader.load())

    return documents

  # -------------------------
  # 2. Chunk Documents
  # -------------------------
  def split_documents(self, documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
      chunk_size = self.chunk_size,
      chunk_overlap = self.chunk_overlap
    )
    return splitter.split_documents(documents)

  # -------------------------
  # 3. Build Vector Store
  # -------------------------
  def build_index(self):
    logger.info("Starting vector index build")

    raw_docs = self.load_documents()

    logger.info(
      "Raw documents loaded",
      extra={"extra_data": {"count": len(raw_docs)}}
    )

    if not raw_docs:
      logger.error("No documents found. Skipping index build.")
      return

    chunks = self.split_documents(raw_docs)

    logger.info(
      "Documents(chunks) after chunking",
      extra={"extra_data": {"count": len(chunks)}}
    )

    if not chunks:
      logger.error("No chunks generated. Skipping index build.")
      return

    self.vectorstore = FAISS.from_documents(
      chunks,
      self.embeding_model
    )

    logger.info("Vector index successfully built")

  # -------------------------
  # 4. Retrieve Relevant Chunks
  # -------------------------
  def retrieve(self, query: str, k: int = 4) -> List[Document]:
    if not self.vectorstore:
      raise RuntimeError("Vector store not initialized. Call build_index() first.")

    return self.vectorstore.similarity_search(query, k = k)