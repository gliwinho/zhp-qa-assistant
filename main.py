from pathlib import Path
from core.rag import chain, data_handler


def main():
    data_handler.save_docs_to_vectorstore((Path.cwd() / "data").glob("*.pdf"))
    db = data_handler.load_docs_from_vectorstore()
    while True:
        question = input('>> ')
        context = db.max_marginal_relevance_search(question, k=3, fetch_k=5)
        output = chain.invoke({
            "context": context,
            "question": question
        })
        print(output["text"])


if __name__ == "__main__":
    main()
