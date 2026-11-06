
import os
import glob
import pandas as pd
from lxml import etree

# Dataset location
DATASET_PATH = "data/raw/MedQuAD"

records = []

# Find all XML files recursively
xml_files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*.xml"),
    recursive=True
)

print(f"Found {len(xml_files)} XML files")

# Process each XML file
for file in xml_files:
    try:
        tree = etree.parse(file)
        root = tree.getroot()

        topic = root.findtext("Focus", default="Unknown")
        source = root.get("source", "Unknown")
        source_url = root.get("url", "")

        # Extract question-answer pairs
        for pair in root.findall(".//QAPair"):

            question_element = pair.find("Question")
            answer_element = pair.find("Answer")

            if question_element is None or answer_element is None:
                continue

            question = "".join(question_element.itertext()).strip()
            answer = "".join(answer_element.itertext()).strip()

            if question and answer:

                records.append({
                    "topic": topic.strip(),
                    "question": question,
                    "answer": answer,
                    "question_type": question_element.get("qtype", "Unknown"),
                    "source": source,
                    "source_url": source_url
                })

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Convert records into DataFrame
df = pd.DataFrame(records)

# Save processed dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/medical_data.csv",
    index=False
)

print("\n========== EXTRACTION RESULTS ==========")
print("XML files:", len(xml_files))
print("Total QA records:", len(df))
print("CSV saved successfully!")

print("\nSample records:")
print(df.head(3).to_string(index=False))
