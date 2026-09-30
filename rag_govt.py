import os
from typing import List, Dict, Any

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

from groq_client import groq_chat

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embed_model(model_name: str = DEFAULT_EMBED_MODEL):
    return HuggingFaceEmbedding(model_name=model_name)


def build_or_load_govt_index(
    pdf_dir: str = "data/govt_pdfs",
    persist_dir: str = "storage/govt_index",
    collection_name: str = "govt_schemes",
) -> VectorStoreIndex:
    os.makedirs(persist_dir, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Try load existing
    try:
        return load_index_from_storage(storage_context=storage_context, embed_model=_get_embed_model())
    except Exception:
        pass

    if not os.path.isdir(pdf_dir):
        raise RuntimeError(f"PDF directory not found: {pdf_dir}")

    documents = SimpleDirectoryReader(pdf_dir, recursive=True).load_data()
    if not documents:
        raise RuntimeError(f"No documents found in {pdf_dir}. Add PDFs first.")

    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=_get_embed_model(),
    )


def govt_rag_answer(
    question: str,
    index: VectorStoreIndex,
    model: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    contexts: List[Dict[str, str]] = []
    for i, n in enumerate(nodes, start=1):
        meta = n.node.metadata or {}
        source = meta.get("file_name") or meta.get("filename") or meta.get("source") or "unknown"
        page = meta.get("page_label") or meta.get("page") or ""
        contexts.append(
            {
                "id": f"S{i}",
                "source": str(source),
                "page": str(page),
                "text": n.node.get_text(),
            }
        )

    context_block = "\n\n".join(
        [f"[{c['id']}] (source={c['source']}, page={c['page']})\n{c['text']}" for c in contexts]
    )

    prompt = f"""
You are a helpful assistant for Indian farmers answering questions about government schemes and services.

RULES (important):
- Answer ONLY using the provided SOURCES. If the sources do not contain the answer, say: "Not found in provided documents".
- Do NOT invent eligibility rules, deadlines, amounts, or websites.
- Provide step-by-step guidance if available in sources.
- Always include citations like [S1], [S2] after relevant sentences.
- End with: "Please verify on the official government portal / local agriculture office."

QUESTION:
{question}

SOURCES:
{context_block}

Now write the answer.
"""
    answer = groq_chat(prompt, model=model)
    return {"answer": answer, "sources": contexts}