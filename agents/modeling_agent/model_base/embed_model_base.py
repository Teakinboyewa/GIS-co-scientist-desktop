from dotenv import load_dotenv
load_dotenv()

import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# All source JSON files to combine
SOURCE_FILES = [
    "sklearn_models.json",
    "spreg_models.json",
    "esda_models.json",
    "mgwr_model.json",
    "spglm_model.json",
]

def combine_model_docs():
    """Load all source JSON files and combine into a single model_docs.json."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    combined = []

    for src_file in SOURCE_FILES:
        path = os.path.join(base_dir, src_file)
        if not os.path.exists(path):
            print(f"  Warning: {src_file} not found, skipping.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            models = json.load(f)
        print(f"  Loaded {len(models)} models from {src_file}")
        combined.extend(models)

    # Save combined file
    out_path = os.path.join(base_dir, "model_docs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\nCombined {len(combined)} models into model_docs.json")
    return combined


def embed_model_base():
    """Build FAISS index from the combined model_docs.json."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(base_dir, "model_docs.json")

    if not os.path.exists(docs_path):
        print("model_docs.json not found. Running combine_model_docs() first...")
        combine_model_docs()

    with open(docs_path, "r", encoding="utf-8") as f:
        model_docs = json.load(f)

    # Build LangChain Documents from the combined list
    docs = []
    for m in model_docs:
        content = m.get("page_content", "")
        if not content:
            continue
        meta = m.get("metadata", {})
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "model_name": meta.get("model_name", ""),
                    "full_class": meta.get("full_class", ""),
                    "type": meta.get("type", ""),
                    "source": meta.get("source", ""),
                }
            )
        )

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(os.path.join(base_dir, "faiss_index"))

    print(f"Indexed {len(docs)} models into FAISS.")


if __name__ == "__main__":
    combine_model_docs()
    embed_model_base()