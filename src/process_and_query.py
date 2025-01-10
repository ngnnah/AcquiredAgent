import os
import numpy as np
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SimpleNodeParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
RAW_DATA_DIR = "./data/raw_transcripts"
PROCESSED_DATA_DIR = "./data/processed_index"
OLLAMA_MODEL = "nhat:latest"
LLM_TIMEOUT = 60.0


def process_transcripts():
    print("Initializing embedding model and LLM...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=LLM_TIMEOUT)

    # Debug: Test embedding model
    test_embedding = Settings.embed_model.get_text_embedding("Test embedding")
    if test_embedding is None:
        print("Error: Embedding model failed to generate test embedding")
    else:
        print(
            f"Debug: Test embedding generated successfully. Shape: {np.array(test_embedding).shape}"
        )

    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"Loading existing index from {PROCESSED_DATA_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=PROCESSED_DATA_DIR)
        index = load_index_from_storage(storage_context)

        existing_docs = set(
            os.path.basename(doc.metadata["file_name"])
            for doc in index.docstore.docs.values()
        )
        all_docs = set(os.listdir(RAW_DATA_DIR))
        new_docs = all_docs - existing_docs

        if new_docs:
            print(f"Found {len(new_docs)} new documents. Updating index...")
            reader = SimpleDirectoryReader(RAW_DATA_DIR, filename_as_id=True)
            documents = [
                doc
                for doc in reader.load_data()
                if os.path.basename(doc.metadata["file_name"]) in new_docs
            ]
            parser = SimpleNodeParser.from_defaults()
            nodes = parser.get_nodes_from_documents(documents)
            index.insert_nodes(nodes)
        else:
            print("No new documents found. Index is up to date.")
    else:
        print(f"Creating new index from {RAW_DATA_DIR}...")
        documents = SimpleDirectoryReader(RAW_DATA_DIR, filename_as_id=True).load_data()
        index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=PROCESSED_DATA_DIR)
    print(
        f"Index {'updated' if os.path.exists(PROCESSED_DATA_DIR) else 'created'} and saved to {PROCESSED_DATA_DIR}"
    )

    print("Debug: Checking if embed_model is set in Settings")
    if hasattr(Settings, "embed_model"):
        print(f"Debug: embed_model is set: {type(Settings.embed_model)}")
    else:
        print("Error: embed_model is not set in Settings")

    return index


def query_index(index, query_text):
    query_engine = index.as_query_engine()

    # Debug: Print query text and check embedding
    print(f"Debug: Query text: {query_text}")
    query_embedding = Settings.embed_model.get_text_embedding(query_text)
    if query_embedding is None:
        print("Error: Failed to generate query embedding")
        return "Error: Unable to process query", []

    # First query for the main answer
    try:
        main_response = query_engine.query(query_text)
    except Exception as e:
        print(f"Error during main query: {str(e)}")
        return f"Error: {str(e)}", []

    # Second query for TLDR and follow-up questions
    followup_query = f"""Based on the following answer to the question "{query_text}", please provide:
1. A brief TLDR summary (2-3 sentences)
2. Three suggested follow-up questions

Answer:
{main_response.response}

Format your response exactly as follows:
TLDR: [Your TLDR here]

Suggested Follow-up Questions:
1. [First follow-up question]
2. [Second follow-up question]
3. [Third follow-up question]
"""

    try:
        followup_response = query_engine.query(followup_query)
    except Exception as e:
        print(f"Error during followup query: {str(e)}")
        followup_response = None

    # Combine the responses
    formatted_response = f"""Detailed Answer:
{main_response.response}

{followup_response.response if followup_response else "Error: Unable to generate follow-up information."}
"""

    # Combine source nodes from both queries
    all_source_nodes = main_response.source_nodes + (
        followup_response.source_nodes if followup_response else []
    )
    # Remove duplicates while preserving order
    unique_source_nodes = []
    seen = set()
    for node in all_source_nodes:
        if node.node.node_id not in seen:
            seen.add(node.node.node_id)
            unique_source_nodes.append(node)

    return formatted_response, unique_source_nodes


def display_index_info(index):
    print("\n--- Index Information ---")
    print(f"Vector store type: {type(index.vector_store)}")
    print(f"Number of documents in the index: {len(index.docstore.docs)}")
    print(f"Embedding dimension: {index.vector_store.dim}")
    print(f"Total vectors: {index.vector_store.num_vectors()}")
    print("---------------------------\n")


def display_conversation_embedding(index, conversation):
    print("\n--- Conversation Embedding ---")
    conversation_text = " ".join(conversation)
    embedding = index.embed_model.get_text_embedding(conversation_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Embedding preview (first 10 values): {embedding[:10]}")
    print(f"Embedding statistics:")
    print(f"  Mean: {np.mean(embedding):.4f}")
    print(f"  Std Dev: {np.std(embedding):.4f}")
    print(f"  Min: {np.min(embedding):.4f}")
    print(f"  Max: {np.max(embedding):.4f}")
    print("-------------------------------\n")


def chat_loop(index):
    print("Welcome to PhiloBot! Type 'exit' to end the conversation.")
    print("Type 'show index' to display current index information.")
    print("Type 'show embedding' to display current conversation embedding.")
    conversation = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "show index":
            display_index_info(index)
            continue
        elif user_input.lower() == "show embedding":
            display_conversation_embedding(index, conversation)
            continue

        conversation.append(user_input)
        response, source_nodes = query_index(index, user_input)
        conversation.append(response)
        print("\nPhiloBot:")
        print(response)

        print("\nSource Nodes:")
        for i, node in enumerate(source_nodes, 1):
            print(f"\n--- Source Node {i} ---")
            print(f"Content: {node.node.text[:200]}...")
            print(f"Source: {node.node.metadata.get('file_name', 'Unknown')}")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    index = process_transcripts()
    print(f"Vector store type: {type(index.vector_store)}")
    print(f"Number of documents in the index: {len(index.docstore.docs)}")

    if index.docstore.docs:
        sample_id = next(iter(index.docstore.docs))
        sample_doc = index.docstore.docs[sample_id]
        print(f"Sample document metadata: {sample_doc.metadata}")
        print(f"Sample document text (first 100 chars): {sample_doc.text[:100]}")
    else:
        print("Warning: No documents found in the index.")

    chat_loop(index)
