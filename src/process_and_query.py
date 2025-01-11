import sqlite3
import pandas as pd
from datetime import datetime
from typing import List
import os
import json
import numpy as np
from datetime import datetime
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
RAW_DATA_DIR = "./data/raw_transcripts"
PROCESSED_DATA_DIR = "./data/processed_index"
METADATA_DB = "./data/episode_metadata.db"
OLLAMA_MODEL = "nhat:latest"
LLM_TIMEOUT = 60.0


def init_metadata_db():
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS episodes
                 (id INTEGER PRIMARY KEY, title TEXT, date DATE, url TEXT)"""
    )
    conn.commit()
    conn.close()


def insert_episode_metadata(title, date, url):
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    parsed_date = parse_date(date)
    c.execute(
        "INSERT OR REPLACE INTO episodes (title, date, url) VALUES (?, ?, ?)",
        (title, parsed_date.strftime("%Y-%m-%d"), url),
    )
    conn.commit()
    conn.close()


def parse_date(date_string):
    try:
        return datetime.strptime(date_string, "%B %d, %Y")
    except ValueError:
        print(
            f"Warning: Unable to parse date '{date_string}'. Using current date instead."
        )
        return datetime.now()


def generate_metadata_summary():
    conn = sqlite3.connect(METADATA_DB)
    df = pd.read_sql_query("SELECT * FROM episodes", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isnull().all():
        print(
            "Warning: All dates are invalid. Please check the date format in your JSON files."
        )
        return "Error: Unable to generate metadata summary due to invalid dates."

    summary = f"""
    Metadata Summary for Acquired FM Episodes:
    
    Total Episodes: {len(df)}
    Date Range: From {df['date'].min().strftime('%B %d, %Y')} to {df['date'].max().strftime('%B %d, %Y')}
    
    Newest Episode: {df.loc[df['date'].idxmax(), 'title']} ({df['date'].max().strftime('%B %d, %Y')})
    Oldest Episode: {df.loc[df['date'].idxmin(), 'title']} ({df['date'].min().strftime('%B %d, %Y')})
    
    5 Most Recent Episodes:
    {df.sort_values('date', ascending=False).head(5)[['title', 'date']].to_string(index=False)}
    
    Episodes by Year:
    {df['date'].dt.year.value_counts().sort_index().to_string()}
    
    This summary provides an overview of the Acquired FM episode collection. For more specific queries,
    you can ask about episodes from a particular year, date range, or any other metadata-related question.
    """
    return summary


class EpisodeMetadata:
    def __init__(self):
        self.episodes = []
        self.oldest_episode = None
        self.latest_episode = None

    def add_episode(self, title, date, url):
        episode_date = datetime.strptime(date, "%B %d, %Y")
        self.episodes.append({"title": title, "date": episode_date, "url": url})

        if not self.oldest_episode or episode_date < self.oldest_episode["date"]:
            self.oldest_episode = {"title": title, "date": episode_date, "url": url}

        if not self.latest_episode or episode_date > self.latest_episode["date"]:
            self.latest_episode = {"title": title, "date": episode_date, "url": url}

    def get_metadata_summary(self):
        return f"""
        Total Episodes: {len(self.episodes)}
        Oldest Episode: {self.oldest_episode['title']} ({self.oldest_episode['date'].strftime('%B %d, %Y')})
        Latest Episode: {self.latest_episode['title']} ({self.latest_episode['date'].strftime('%B %d, %Y')})
        """


episode_metadata = EpisodeMetadata()


def load_json_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as file:
                data = json.load(file)

                # Add episode to metadata
                episode_metadata.add_episode(data["title"], data["date"], data["url"])

                # Create metadata document
                metadata_text = f"Title: {data['title']}\n"
                metadata_text += f"Date: {data['date']}\n"
                metadata_text += f"Heading: {data['heading']}\n"
                metadata_text += f"URL: {data['url']}\n"
                metadata_text += f"Description: {data['description']}\n"

                metadata_doc = Document(
                    text=metadata_text,
                    metadata={
                        "type": "metadata",
                        "title": data["title"],
                        "date": data["date"],
                        "file_name": filename,
                    },
                )
                documents.append(metadata_doc)

                # Create transcript document
                transcript_doc = Document(
                    text=data["transcript"],
                    metadata={
                        "type": "transcript",
                        "title": data["title"],
                        "date": data["date"],
                        "file_name": filename,
                    },
                )
                documents.append(transcript_doc)

    return documents


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_index, metadata_db):
        self.vector_index = vector_index
        self.metadata_db = metadata_db

    def _retrieve(self, query_str: str, **kwargs) -> List[NodeWithScore]:
        # Always include the metadata summary
        metadata_summary = generate_metadata_summary()
        metadata_node = NodeWithScore(
            node=Document(
                text=metadata_summary, metadata={"source": "metadata_summary"}
            ),
            score=1.0,
        )

        # Retrieve from vector index
        vector_results = self.vector_index.as_retriever().retrieve(query_str)

        # Combine results, putting metadata summary first
        return [metadata_node] + vector_results

    def query_metadata(self, query_str: str) -> str:
        conn = sqlite3.connect(self.metadata_db)

        try:
            # Use the LLM to generate an SQL query based on the natural language question
            sql_query = Settings.llm.complete(
                f"Generate an SQL query to answer the following question about podcast episodes: {query_str}\nSQL query:"
            ).text

            # Execute the generated SQL query
            df = pd.read_sql_query(sql_query, conn)

            # Format the results
            if len(df) > 0:
                return f"Query results:\n{df.to_string()}"
            else:
                return "No results found for the given query."
        except Exception as e:
            return f"Error executing query: {str(e)}"
        finally:
            conn.close()


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
        return None
    else:
        print(
            f"Debug: Test embedding generated successfully. Shape: {np.array(test_embedding).shape}"
        )

    try:
        # Initialize metadata database
        init_metadata_db()

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        if os.path.exists(os.path.join(PROCESSED_DATA_DIR, "docstore.json")):
            print(f"Loading existing index from {PROCESSED_DATA_DIR}...")
            storage_context = StorageContext.from_defaults(
                persist_dir=PROCESSED_DATA_DIR
            )
            index = load_index_from_storage(storage_context)

            existing_docs = set(
                doc.metadata["file_name"]
                for doc in index.docstore.docs.values()
                if "file_name" in doc.metadata
            )

            all_docs = set(os.listdir(RAW_DATA_DIR))
            new_docs = all_docs - existing_docs

            if new_docs:
                print(f"Found {len(new_docs)} new documents. Updating index...")
                new_documents = load_json_files(RAW_DATA_DIR)
                for doc in new_documents:
                    if doc.metadata.get("type") == "metadata":
                        # Insert metadata into SQLite database
                        insert_episode_metadata(
                            doc.metadata["title"],
                            doc.metadata["date"],
                            doc.metadata.get("url", ""),
                        )

                parser = SimpleNodeParser.from_defaults()
                nodes = parser.get_nodes_from_documents(new_documents)
                index.insert_nodes(nodes)

                # Generate and update metadata summary
                metadata_summary = generate_metadata_summary()
                summary_node = Document(
                    text=metadata_summary, metadata={"type": "metadata_summary"}
                )
                index.insert(summary_node)

                index.storage_context.persist(persist_dir=PROCESSED_DATA_DIR)
            else:
                print("No new documents found. Index is up to date.")
        else:
            print(f"Creating new index from {RAW_DATA_DIR}...")
            documents = load_json_files(RAW_DATA_DIR)
            for doc in documents:
                if doc.metadata.get("type") == "metadata":
                    # Insert metadata into SQLite database
                    insert_episode_metadata(
                        doc.metadata["title"],
                        doc.metadata["date"],
                        doc.metadata.get("url", ""),
                    )

            # Generate metadata summary
            metadata_summary = generate_metadata_summary()
            summary_doc = Document(
                text=metadata_summary, metadata={"type": "metadata_summary"}
            )
            documents.append(summary_doc)

            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=PROCESSED_DATA_DIR)

        print(
            f"Index {'updated' if os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'docstore.json')) else 'created'} and saved to {PROCESSED_DATA_DIR}"
        )

        print("Debug: Checking if embed_model is set in Settings")
        if hasattr(Settings, "embed_model"):
            print(f"Debug: embed_model is set: {type(Settings.embed_model)}")
        else:
            print("Error: embed_model is not set in Settings")

        return index

    except Exception as e:
        print(f"Error processing transcripts: {e}")
        import traceback

        traceback.print_exc()
        return None


def query_index(index, query_text):
    print(f"Type of index: {type(index)}")
    hybrid_retriever = HybridRetriever(index, METADATA_DB)

    # Create a response synthesizer
    response_synthesizer = TreeSummarize(verbose=True)

    # Create the query engine directly
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
    )

    # Debug: check embedding
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
    print(
        """
Welcome to AcquiredAgent, your AI guide to the world of Acquired FM!

🚀 Dive into the strategies of tech giants and unicorns
📚 Access insights from 3-4 hour "conversational audiobooks"
🎙️ Explore episodes featuring titans like NVIDIA, Berkshire Hathaway, and Meta

I'm here to help you navigate the #1 Technology show's vast knowledge base. 
What would you like to know about great companies and their success stories?

(Type 'exit' when you're ready to conclude our chat.)
(Type 'show index' to display current index information.)
(Type 'show embedding' to display current conversation embedding.)
"""
    )
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

    # sample convo: Yay!
    # You: how many episodes and which one is newest and oldest
    # Detailed Answer:
    # There are a total of 288 episodes. The oldest episode is Pixar from October 15, 2015, and the latest episode is Mars Inc. (the chocolate story) from December 15, 2024.
    # TLDR: There are 288 episodes, with Pixar as the oldest episode from October 15, 2015, and Mars Inc. (the chocolate story) being the latest one from December 15, 2024.
    chat_loop(index)
