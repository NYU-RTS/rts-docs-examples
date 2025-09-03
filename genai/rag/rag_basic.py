# Adapted from https://docling-project.github.io/docling/examples/rag_milvus

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from typing import List
from argparse import ArgumentParser
from portkey_ai import Portkey
from pymilvus import MilvusClient
from dataclasses import dataclass
from tqdm import tqdm
import os
import sys


@dataclass
class RAG_config:
    milvus_client: MilvusClient
    portkey_client: Portkey
    milvus_collection_name: str
    dimensions: int


def get_chunks_from_webpage(url) -> List[str]:
    source = url
    converter = DocumentConverter()
    chunker = HybridChunker()
    doc = converter.convert(source).document
    text_chunks = [chunk.text for chunk in chunker.chunk(doc)]
    for chunk in text_chunks:
        print("\n--------------------------------------------------------------------")
        print(chunk)
    return text_chunks


def populate_vector_db(text_chunks: List[str], rag_config: RAG_config) -> None:
    data = []
    for i, chunk in enumerate(tqdm(text_chunks, desc="Processing chunks")):
        response = rag_config.portkey_client.embeddings.create(
            model="@vertexai/gemini-embedding-001",
            input=chunk,
            encoding_format="float",
        )
        embedding_vector = response["data"][0].embedding
        data.append({"id": i, "vector": embedding_vector, "text": chunk})

    rag_config.milvus_client.insert(
        collection_name=rag_config.milvus_collection_name, data=data
    )

    return


parser = ArgumentParser(
    prog="rag_basic",
    description="Basic demonstration of Retrieval-Augmented Generation",
)
parser.add_argument(
    "url",
    help="URL of website to store in vector db after\
        parsing and chunking",
)
parser.add_argument(
    "query", help="query to LLM which can benefit from information at URL"
)

if __name__ == "__main__":
    args = parser.parse_args()

    try:
        portkey_api_key = os.environ["PORTKEY_API_KEY"]
    except KeyError:
        print(
            "PORTKEY_API_KEY has not been set.\
            Please set and re-run!"
        )
        sys.exit()

    rag_config = RAG_config(
        portkey_client=Portkey(
            base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
            api_key=portkey_api_key,
        ),
        milvus_client=MilvusClient("./rag_demo.db"),
        milvus_collection_name="rag_demo",
        dimensions=256,
    )

    # Drop existing collection with the same name if it exists
    if rag_config.milvus_client.has_collection(rag_config.milvus_collection_name):
        rag_config.milvus_client.drop_collection(rag_config.milvus_collection_name)

    rag_config.milvus_client.create_collection(
        collection_name=rag_config.milvus_collection_name,
        dimension=rag_config.dimensions,  # Default embedding length for gemini-embedding-001
    )

    text_chunks = get_chunks_from_webpage(args.url)

    populate_vector_db(text_chunks, rag_config)

    # Embed Query
    response = rag_config.portkey_client.embeddings.create(
        model="@vertexai/gemini-embedding-001",
        input=args.query,
        encoding_format="float",
        dimensions=rag_config.dimensions,
    )
    query_embedding_vecor = response["data"][0].embedding
    print("----------------------------------------------------------------")
    print("Query embedding vector (first 5 dims) is: ", query_embedding_vecor[:5])

    search_res = rag_config.milvus_client.search(
        collection_name=rag_config.milvus_collection_name,
        data=[query_embedding_vecor],
        limit=3,
        output_fields=["text"],
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]

    print("----------------------------------------------------------------")
    print("Retreived chunks and similarity scores:\n")
    for retrieved_line_with_distance in retrieved_lines_with_distances:
        print(retrieved_line_with_distance)

    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    completion = rag_config.portkey_client.chat.completions.create(
        model="@vertexai/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are not a helpful assistant"},
            {
                "role": "user",
                "content": args.query,
            },
        ],
    )

    print("----------------------------------------------------------------")
    print("Generated response from LLM without additional context is:\n")
    print(completion.choices[0]["message"]["content"])

    completion = rag_config.portkey_client.chat.completions.create(
        model="@vertexai/gemini-2.5-flash",
        messages=[
            {
                "role": "system",
                "content": """Human: You are an AI assistant. You are able to
                    find answers to the questions from the contextual passage
                    snippets provided.""",
            },
            {
                "role": "user",
                "content": f"""Use the following pieces of information enclosed
                    in <context>  tags to provide an answer to the question
                    enclosed in <question> tags.
                    <context> {context} </context>
                    <question> {args.query} </question> """,
            },
        ],
    )

    print("---------------------------------------------------------------")
    print("Generated response from LLM with additional context is:\n")
    print(completion.choices[0]["message"]["content"])
