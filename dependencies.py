"""
FastAPI dependency providers.

This module defines injectable factory functions used with FastAPI's
:func:`~fastapi.Depends` system.  Each function instantiates and returns a
service or store object that route handlers can receive as a typed parameter
without managing lifecycle themselves.

Using dependency injection here keeps route handlers decoupled from
concrete implementations and makes unit testing straightforward — simply
override the dependency in tests to inject a mock.
"""

from services.ingest_service import IngestService
from services.graph_services import GraphService
from core.vector_store import NeonVectorStore


def get_ingest_service() -> IngestService:
    """Provide an :class:`~services.ingest_service.IngestService` instance.

    Returns:
        A new :class:`~services.ingest_service.IngestService` object ready
        for use in an upload route handler.
    """
    return IngestService()


def get_graph_service() -> GraphService:
    """Provide a :class:`~services.graph_services.GraphService` instance.

    Returns:
        A new :class:`~services.graph_services.GraphService` object ready
        for use in a chat route handler.
    """
    return GraphService()


def get_vector_store() -> NeonVectorStore:
    """Provide a :class:`~core.vector_store.NeonVectorStore` instance.

    Returns:
        A new :class:`~core.vector_store.NeonVectorStore` object ready for
        database operations in route handlers (listing, deleting documents,
        health checks).
    """
    return NeonVectorStore()