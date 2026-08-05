from app.retriever import Retriever
from app.llm import LLM


def main():

    retriever = Retriever()

    llm = LLM()

    while True:

        question = input("\nAsk Question: ")

        if question.lower() == "exit":
            break

        results = retriever.hybrid_search(
            query=question,
            top_k=3
        )

        if not results:

            print("\nNo relevant document found.")
            continue

        context = ""

        print("\nRetrieved Chunks\n")

        for i, result in enumerate(results, start=1):

            print("=" * 60)
            print(f"Rank : {i}")
            print(f"Score: {result.score:.4f}")
            print(f"Page : {result.payload['page']}")
            print("-" * 60)
            print(result.payload["text"])
            print()

            context += result.payload["text"] + "\n\n"

        answer = llm.answer_question(
            question,
            context
        )

        print("\nAnswer\n")
        print(answer)


if __name__ == "__main__":
    main()