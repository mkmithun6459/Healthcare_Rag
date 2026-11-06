
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load extracted medical dataset
df = pd.read_csv("data/processed/medical_data.csv")

print("Total records:", len(df))

# Load embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="medical_knowledge"
)

# Prepare documents
documents = []
metadatas = []
ids = []

for index, row in df.iterrows():

    text = (
        f"Medical Topic: {row['topic']}\n"
        f"Question: {row['question']}\n"
        f"Answer: {row['answer']}"
    )

    documents.append(text)

    metadatas.append({
        "topic": str(row["topic"]),
        "question_type": str(row["question_type"]),
        "source": str(row["source"]),
        "source_url": str(row["source_url"])
    })

    ids.append(str(index))

# Generate embeddings and store in batches
batch_size = 128

for start in range(0, len(documents), batch_size):

    end = start + batch_size

    batch_docs = documents[start:end]

    embeddings = model.encode(
        batch_docs,
        show_progress_bar=False
    ).tolist()

    collection.upsert(
        ids=ids[start:end],
        documents=batch_docs,
        embeddings=embeddings,
        metadatas=metadatas[start:end]
    )

    print(f"Processed {min(end, len(documents))}/{len(documents)}")

print("\nEmbedding generation completed!")
print("Total vectors:", collection.count())
