"""
Chat History & Session Management Store.

This module provides :class:`ChatHistoryStore`, a PostgreSQL-backed data layer
for persisting multi-turn conversation messages, running session summaries,
and session lifecycle queries.
"""

import os 
import json
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()


class ChatHistoryStore:
    """PostgreSQL-backed store for multi-turn chat memory and session summaries.

    All methods are class-methods; no instantiation is required.
    """

    DATABASE_URL = os.getenv("DATABASE_URL")

    @classmethod
    def connect(cls) -> psycopg.Connection:
        """Open and return a database connection to Neon PostgreSQL.

        Returns:
            An open :class:`psycopg.Connection` object.

        Raises:
            ValueError: If ``DATABASE_URL`` environment variable is not set.
        """
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL is missing from .env")
        return psycopg.connect(cls.DATABASE_URL)

    @classmethod
    def save_message(
        cls,
        session_id: str,
        role: str,
        content: str,
        metadata: dict = None
    ) -> None:
        """Save a single dialogue turn (user query or assistant response) to the database.

        Args:
            session_id: Unique string identifier for the chat session.
            role: The author role (e.g. ``"user"`` or ``"assistant"``).
            content: The text content of the message.
            metadata: Optional dictionary of metadata to store as JSON.
        """
        meta_json = json.dumps(metadata or {})
        with cls.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages (
                        session_id, role, content, metadata
                    )
                    VALUES (%s, %s, %s, %s);
                    """,
                    (session_id, role, content, meta_json)
                )

    @classmethod
    def get_last_20_message(
        cls,
        session_id: str
    ) -> list[dict]:
        """Retrieve the last 20 messages for a session in chronological order.

        Args:
            session_id: Unique string identifier for the chat session.

        Returns:
            A list of dicts with keys ``"role"`` and ``"content"`` ordered from oldest to newest.
        """
        with cls.connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT 20;
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
        return list(reversed(rows))
    
    @classmethod
    def get_summary(
        cls,
        session_id: str
    ) -> str:
        """Retrieve the running text summary of older messages for a session.

        Args:
            session_id: Unique string identifier for the chat session.

        Returns:
            The summary text string, or an empty string if no summary exists yet.
        """
        with cls.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT summary_text
                    FROM chat_summaries WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
        return row[0] if row else ""

    @classmethod
    def update_summary(
        cls,
        session_id: str,
        new_summary: str
    ) -> None:
        """Upsert (insert or update) the running summary text for a session.

        Args:
            session_id: Unique string identifier for the chat session.
            new_summary: The updated summary text string.
        """
        with cls.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_summaries
                    (session_id, summary_text, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (session_id)
                    DO UPDATE SET summary_text = EXCLUDED.summary_text, updated_at = NOW();
                    """,
                    (session_id, new_summary),
                )

    # ------------------------------------------------------------------ #
    #  Session Lifecycle Management                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def list_sessions(cls) -> list[dict]:
        """Retrieve a list of all unique chat sessions with their latest activity timestamp.

        Returns:
            A list of dicts containing ``"session_id"`` and ``"last_active_at"``.
        """
        with cls.connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT session_id, MAX(created_at) as last_active_at
                    FROM chat_messages
                    GROUP BY session_id
                    ORDER BY last_active_at DESC;
                    """
                )
                return cur.fetchall()

    @classmethod
    def delete_session(cls, session_id: str) -> None:
        """Delete all messages and summary data associated with a specific session ID.

        Args:
            session_id: Unique string identifier for the chat session.
        """
        with cls.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_messages WHERE session_id = %s;", (session_id,))
                cur.execute("DELETE FROM chat_summaries WHERE session_id = %s;", (session_id,))

    @classmethod
    def get_all_session_messages(cls, session_id: str) -> list[dict]:
        """Retrieve the complete message history log for a session in chronological order.

        Args:
            session_id: Unique string identifier for the chat session.

        Returns:
            A list of dicts containing ``"role"``, ``"content"``, and ``"created_at"``.
        """
        with cls.connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC;
                    """,
                    (session_id,),
                )
                return cur.fetchall()