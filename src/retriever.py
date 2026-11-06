
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="medical_knowledge"
)

MAX_DISTANCE = 1.0


def retrieve_medical_information(
    query,
    top_k=5,
    max_distance=MAX_DISTANCE
):

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_documents = []
    seen_documents = set()

    for i, document in enumerate(results["documents"][0]):

        distance = results["distances"][0][i]

        # Reject weak matches
        if distance > max_distance:
            continue

        # Remove duplicate documents
        if document in seen_documents:
            continue

        seen_documents.add(document)

        retrieved_documents.append({
            "document": document,
            "metadata": results["metadatas"][0][i],
            "distance": distance
        })

    return retrieved_documents


if __name__ == "__main__":

    question = input("Ask a medical question: ")

    results = retrieve_medical_information(question)

    if not results:
        print("\nNo sufficiently relevant medical information found.")
    else:
        for i, result in enumerate(results, 1):

            print(f"\n{'='*60}")
            print(f"RESULT {i}")
            print(f"{'='*60}")

            print("Topic:", result["metadata"]["topic"])
            print("Source:", result["metadata"]["source"])
            print("Distance:", round(result["distance"], 4))

            print("\nRetrieved information:")
            print(result["document"])
