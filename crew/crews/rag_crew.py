import sys, os

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from config.openrouter_settings import RouterModel, RouterConfig
from crew.tools.rag_tool import search_pdf

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "test-rag"

model = RouterModel.MINIMAX25.value
router_config = RouterConfig.config(model)

llm = LLM(
    # model = "openrouter/cohere/north-mini-code:free",
    # model = "openrouter/openrouter/free",
    model = f"openrouter/{model}",
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key = os.environ.get("OPENROUTER_API_KEY"),
    extra_body= router_config
)

retriever_agent = Agent(
    role = "Document Retriever",
    goal = "Find the most relevant chunks from the PDF document "
    "for the given question",
    backstory = "You are an expert at searching documents and "
    "extracting the right information.",
    tools = [search_pdf],
    llm = llm,
    verbose = True
)

answer_agent = Agent(
    role = "Answer Synthesizer",
    goal = "Give a clear, accurate answer using only the "
    "retrieved document chunks",
    backstory = "You are an expert at reading source material and "
    "giving grounded, concise answers.",
    llm = llm,
    verbose = True
)

verifier_agent = Agent(
    role = "Answer Verifier",
    goal = "Verify that the answer is fully supported by the "
    "retrieved chunks. Correct any unsupported claims.",
    backstory = "You are a strict fact-checker. Who never allow "
    "answers that go beyond the source chunks say.",
    llm = llm,
    verbose = True
)

retrieve_task = Task(
    description = "Search the PDF documents to find relevant "
    "information for this question: {question}",
    expected_output = "A list of relevant text chunks from the "
    "PDF documents.",
    agent = retriever_agent
)

answer_task = Task(
    description = "Using the retrieved chunks, answer this question: "
    "{question} Use only what the chunks say.",
    expected_output = "A clear, concise answer grounded strictly "
    "in the retrieved document chunks.",
    agent = answer_agent
)

verify_task = Task(
    description= ("Check if the answer to '{question}' is fully "\
    "supported by the retrieved chunks. Remove or correct any claims "\
    "not found in the chunks. Output the final answer."
    ),
    expected_output= "A verified, corrected answer grounded strictly "\
    "in the source chunks.",
    agent = verifier_agent,
    context = [retrieve_task, answer_task]
)

rag_crew = Crew(
    agents = [retriever_agent, answer_agent, verifier_agent],
    tasks = [retrieve_task, answer_task, verify_task],
    process = Process.sequential,
    verbose = True
)
