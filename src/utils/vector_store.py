# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
Vector store utilities using ChromaDB for persistent embeddings storage.

This module provides a class for creating and managing vector databases
using ChromaDB with LangChain integration.
"""

import os
from pathlib import Path
from typing import List, Optional, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma


class VectorStore:
    """
    A class to manage vector databases using ChromaDB.

    This class provides methods to create, populate, and query vector stores
    with persistent storage in the embeddings directory.

    Attributes:
        persist_directory: Path where the vector database is stored.
        collection_name: Name of the ChromaDB collection.
        embedding_function: The embedding model to use for vectorization.
        vector_store: The ChromaDB vector store instance.

    Example:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> embeddings = OpenAIEmbeddings()
        >>> store = VectorStore(
        ...     collection_name="my_docs",
        ...     embedding_function=embeddings
        ... )
        >>> store.add_documents([Document(page_content="Hello world")])
        >>> results = store.similarity_search("greeting", k=1)
    """

    DEFAULT_EMBEDDINGS_PATH = Path(__file__).parent.parent.parent / "data" / "embeddings"

    def __init__(
        self,
        collection_name: str,
        embedding_function: Embeddings,
        persist_directory: Optional[str] = None,
    ) -> None:
        """
        Initialize the VectorStore.

        Args:
            collection_name: Name for the ChromaDB collection.
            embedding_function: LangChain embedding model for vectorization.
            persist_directory: Optional custom path for database storage.
                Defaults to data/embeddings/{collection_name}.
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        if persist_directory:
            self.persist_directory = Path(persist_directory)
        else:
            self.persist_directory = self.DEFAULT_EMBEDDINGS_PATH / collection_name

        self._ensure_directory_exists()
        self.vector_store = self._initialize_store()

    def _ensure_directory_exists(self) -> None:
        """Create the persist directory if it doesn't exist."""
        os.makedirs(self.persist_directory, exist_ok=True)

    def _initialize_store(self) -> Chroma:
        """
        Initialize or load an existing ChromaDB vector store.

        Returns:
            Chroma: The initialized vector store instance.
        """
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=str(self.persist_directory),
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects to add.
            ids: Optional list of unique identifiers for each document.

        Returns:
            List of document IDs that were added.
        """
        return self.vector_store.add_documents(documents=documents, ids=ids)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add raw texts to the vector store.

        Args:
            texts: List of text strings to add.
            metadatas: Optional list of metadata dicts for each text.
            ids: Optional list of unique identifiers for each text.

        Returns:
            List of document IDs that were added.
        """
        return self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Search for documents similar to the query.

        Args:
            query: The search query string.
            k: Number of results to return. Defaults to 4.
            filter: Optional metadata filter for the search.

        Returns:
            List of Document objects most similar to the query.
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        """
        Search for documents with similarity scores.

        Args:
            query: The search query string.
            k: Number of results to return. Defaults to 4.
            filter: Optional metadata filter for the search.

        Returns:
            List of tuples containing (Document, similarity_score).
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete.
        """
        self.vector_store.delete(ids=ids)

    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents stored in the vector store.
        """
        return self.vector_store._collection.count()

    def as_retriever(self, **kwargs: Any) -> Any:
        """
        Convert the vector store to a LangChain retriever.

        Args:
            **kwargs: Additional arguments passed to the retriever.

        Returns:
            A LangChain retriever instance.
        """
        return self.vector_store.as_retriever(**kwargs)
