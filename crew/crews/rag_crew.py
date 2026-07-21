"""
CrewAI multi-agent RAG team — CLI-only collaborative Q&A pipeline.

This module builds and wires together a three-agent CrewAI crew that performs
grounded question-answering over ingested PDF documents.

Agents:
    retriever_agent:  Searches the PDF vector database using the custom
                      ``search_pdf`` tool backed by :class:`~services.graph_services.GraphService`.
    answer_agent:     Synthesises a clear, concise answer strictly from the
                      retrieved chunks.
    verifier_agent:   Fact-checks the draft answer against the source chunks
                      and corrects any unsupported claims before final output.

Workflow (sequential):
    Task 1 (retrieve) → Task 2 (answer) → Task 3 (verify) → Final Answer

Environment variables:
    OPENROUTER_API_KEY: API key for the OpenRouter LLM backend.
    LANGSMITH_API_KEY:  Optional key for LangSmith tracing.
"""

import sys
import os

# Add project root to sys.path so imports resolve regardless of working directory.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config.openrouter_settings import RouterModel, RouterConfig
from crew.tools.rag_tool import search_pdf

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

# ------------------------------------------------------------------ #
#  Optional LangSmith tracing                                         #
# ------------------------------------------------------------------ #

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "test-rag"

# ------------------------------------------------------------------ #
#  LLM configuration                                                   #
# ------------------------------------------------------------------ #

#: Primary model for all three agents, routed through OpenRouter.
model = RouterModel.MINIMAX25.value
router_config = RouterConfig.config(model)

llm = LLM(
    model=f"openrouter/{model}",
    base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    extra_body=router_config,
)

# ------------------------------------------------------------------ #
#  Agents                                                              #
# ------------------------------------------------------------------ #

retriever_agent = Agent(
    role="Document Retriever",
    goal=(
        "Find the most relevant chunks from the PDF document "
        "for the given question."
    ),
    backstory=(
        "You are an expert at searching documents and "
        "extracting the right information."
    ),
    tools=[search_pdf],
    llm=llm,
    verbose=True,
)

answer_agent = Agent(
    role="Answer Synthesizer",
    goal=(
        "Give a clear, accurate answer using only the "
        "retrieved document chunks."
    ),
    backstory=(
        "You are an expert at reading source material and "
        "giving grounded, concise answers."
    ),
    llm=llm,
    verbose=True,
)

verifier_agent = Agent(
    role="Answer Verifier",
    goal=(
        "Verify that the answer is fully supported by the "
        "retrieved chunks. Correct any unsupported claims."
    ),
    backstory=(
        "You are a strict fact-checker who never allows "
        "answers that go beyond what the source chunks say."
    ),
    llm=llm,
    verbose=True,
)

# ------------------------------------------------------------------ #
#  Tasks                                                               #
# ------------------------------------------------------------------ #

retrieve_task = Task(
    description=(
        "Search the PDF documents to find relevant "
        "information for this question: {question}"
    ),
    expected_output=(
        "A list of relevant text chunks from the PDF documents."
    ),
    agent=retriever_agent,
)

answer_task = Task(
    description=(
        "Using the retrieved chunks, answer this question: "
        "{question}. Use only what the chunks say."
    ),
    expected_output=(
        "A clear, concise answer grounded strictly "
        "in the retrieved document chunks."
    ),
    agent=answer_agent,
)

verify_task = Task(
    description=(
        "Check if the answer to '{question}' is fully "
        "supported by the retrieved chunks. Remove or correct any claims "
        "not found in the chunks. Output the final answer."
    ),
    expected_output=(
        "A verified, corrected answer grounded strictly "
        "in the source chunks."
    ),
    agent=verifier_agent,
    # Verifier receives the output of both prior tasks as context.
    context=[retrieve_task, answer_task],
)

# ------------------------------------------------------------------ #
#  Crew assembly                                                       #
# ------------------------------------------------------------------ #

#: Compiled sequential crew — import this object in the runner script.
rag_crew = Crew(
    agents=[retriever_agent, answer_agent, verifier_agent],
    tasks=[retrieve_task, answer_task, verify_task],
    process=Process.sequential,
    verbose=True,
)
