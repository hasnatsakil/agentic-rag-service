"""
Health check route — ``GET /health``.

Provides a lightweight liveness and readiness probe endpoint that verifies
the API server is running and that the Neon database connection is healthy.
Intended for use by load balancers, uptime monitors, and container
orchestration systems (e.g. Render health checks).
"""

from fastapi import APIRouter, Depends
from dependencies import get_vector_store
from core.vector_store import NeonVectorStore

router = APIRouter(tags=["Health"])


@router.get("/health")
def health(
    vector_store: NeonVectorStore = Depends(get_vector_store),
):
    """Check API and database health.

    Performs a lightweight database probe (``list_documents``) to confirm
    the Neon connection is active.  Returns ``"ok"`` when both the API and
    database are healthy, or ``"degraded"`` with an error description if
    the database is unreachable.

    Args:
        vector_store: Injected :class:`~core.vector_store.NeonVectorStore`
            (via :func:`~dependencies.get_vector_store`).

    Returns:
        A JSON object with at minimum ``"status"`` and ``"database"`` keys.
        On failure, an additional ``"error"`` key contains the exception
        message.

    Example responses::

        # Healthy
        {"status": "ok", "database": "connected"}

        # Degraded
        {"status": "degraded", "database": "disconnected", "error": "..."}
    """
    try:
        # Ping the DB with a lightweight query to verify connectivity.
        vector_store.list_documents()
        return {
            "status": "ok",
            "database": "connected",
        }
    except Exception as e:
        return {
            "status": "degraded",
            "database": "disconnected",
            "error": str(e),
        }
