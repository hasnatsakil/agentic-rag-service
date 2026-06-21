import sys, os

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from crew.crews.rag_crew import rag_crew

def main():
    question = input("Ask a question about your PDF: ")

    print("\n🚀 Starting RAG Crew...\n")

    result = rag_crew.kickoff(inputs={"question": question})

    print("\n========== FINAL RESULT ==========")
    print(result)
    print("===================================")

if __name__ == "__main__":
    main()