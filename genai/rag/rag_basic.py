# Adapted from https://docling-project.github.io/docling/examples/rag_milvus

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
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


def get_chunks_from_webpage(url) -> List[str]:
    source = url
    converter = DocumentConverter()
    chunker = HierarchicalChunker()
    doc = converter.convert(source).document
    text_chunks = [chunk.text for chunk in chunker.chunk(doc)]
    return text_chunks


def populate_vector_db(text_chunks: List[str], rag_config: RAG_config) -> None:
    data = []
    for i, chunk in enumerate(tqdm(text_chunks, desc="Processing chunks")):
        response = rag_config.portkey_client.embeddings.create(
            model="gemini-embedding-001",
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
        portkey_virtual_key = os.environ["PORTKEY_VIRTUAL_KEY"]
    except KeyError:
        print(
            "PORTKEY_API_KEY or PORTKEY_VIRTUAL_KEY has not been set.\
            Please set and re-run!"
        )
        sys.exit()

    rag_config = RAG_config(
        portkey_client=Portkey(
            base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
            api_key=portkey_api_key,
            virtual_key=portkey_virtual_key,
        ),
        milvus_client=MilvusClient("./rag_demo.db"),
        milvus_collection_name="rag_demo",
    )

    # Drop existing collection with the same name if it exists
    if rag_config.milvus_client.has_collection(rag_config.milvus_collection_name):
        rag_config.milvus_client.drop_collection(rag_config.milvus_collection_name)

    rag_config.milvus_client.create_collection(
        collection_name=rag_config.milvus_collection_name,
        dimension=3072,  # Default embedding length for gemini-embedding-001
    )

    text_chunks = get_chunks_from_webpage(args.url)

    populate_vector_db(text_chunks, rag_config)

    # Embed Query
    response = rag_config.portkey_client.embeddings.create(
        model="gemini-embedding-001",
        input=args.query,
        encoding_format="float",
    )
    query_embedding_vecor = response["data"][0].embedding
    print("Query embedding vector (first 10 dims) is: ", query_embedding_vecor[:10])
    print("----------------------------------------------------------------\n")

    search_res = rag_config.milvus_client.search(
        collection_name=rag_config.milvus_collection_name,
        data=[query_embedding_vecor],
        limit=3,
        output_fields=["text"],
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]

    print("Retreived chunks and similarity scores:")
    for retrieved_line_with_distance in retrieved_lines_with_distances:
        print(retrieved_line_with_distance)
    print("----------------------------------------------------------------\n")

    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    completion = rag_config.portkey_client.chat.completions.create(
        model="gemini-2.5-flash-preview-05-20",
        temperature=2.0,
        messages=[
            {"role": "system", "content": "You are not a helpful assistant"},
            {
                "role": "user",
                "content": args.query,
            },
        ],
    )

    print("Generated response from LLM without additional context is:")
    print(completion.choices[0]["message"]["content"])
    print("----------------------------------------------------------------\n")

    completion = rag_config.portkey_client.chat.completions.create(
        model="gemini-2.5-flash-preview-05-20",
        temperature=2.0,
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

    print("Generated response from LLM with additional context is:")
    print(completion.choices[0]["message"]["content"])
    print("----------------------------------------------------------------\n")
