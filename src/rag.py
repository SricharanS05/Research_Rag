"""
rag.py

Ties retrieval to generation. Builds a strictly-grounded prompt from
retrieved chunks, calls the Groq LLM, and formats citations from the
actual chunks that were used.

The LLM is explicitly instructed to answer ONLY from the provided context
and to say so clearly when the answer isn't present, so the system never
silently falls back on knowledge.RAG Stands for retrieval augmented generation
"""

from typing import List, Dict, Optional

from groq import Groq

from src import config
from src import retriever
from src import vector_store

NOT_FOUND_MESSAGE = "I could not find this information in the uploaded research papers."

_SYSTEM_PROMPT = """You are a research-paper assistant. You must answer questions
using ONLY the context passages provided to you below. Each passage is labeled
with its source file and page number.

Strict rules:
1. Do not use any knowledge you have from outside the provided context.
2. Do not invent, guess, or infer citations, page numbers, authors, datasets,
   or results that are not explicitly present in the context.
3. If the context does not contain the answer, respond exactly with:
   "I could not find this information in the uploaded research papers."
4. Keep answers precise, factual, and directly tied to the given passages.
5. When comparing multiple papers, clearly attribute each fact to its source file.
6. Do not mistake authors cited within the paper's references for the actual
   authors of the paper itself unless the context clearly identifies them as such.
"""


class RAGError(Exception):
    """Raised for configuration or LLM-call failures."""
    pass


def _get_groq_client() -> Groq:
    if not config.is_groq_configured():
        raise RAGError(
            "GROQ_API_KEY is not set. Please add it to your .env file "
            "(see .env.example) before asking questions."
        )
    return Groq(api_key=config.GROQ_API_KEY)


def _format_context(chunks: List[Dict]) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[Source: {chunk['filename']} | Page {chunk['page_number']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def format_citations(chunks: List[Dict]) -> List[str]:
    """De-duplicated, human-readable citation lines from the chunks actually used."""
    seen = set()
    citations = []
    for chunk in chunks:
        key = (chunk["filename"], chunk["page_number"])
        if key in seen:
            continue
        seen.add(key)
        citations.append(f"📄 {chunk['filename']} — Page {chunk['page_number']}")
    return citations


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    client = _get_groq_client()
    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        raise RAGError(f"The language model request failed: {exc}")

    return response.choices[0].message.content.strip()


def answer_question(question: str, doc_ids: Optional[List[str]] = None) -> Dict:
    """
    Full RAG pipeline for a single question:
    retrieve -> build grounded prompt -> generate -> attach citations.

    Returns: {"answer": str, "citations": List[str], "chunks_used": List[Dict]}
    """
    if not question or not question.strip():
        raise RAGError("Please enter a question before asking.")

    if not doc_ids:
        raise RAGError("No papers are selected. Please upload and select at least one paper.")

    chunks = retriever.retrieve(question, doc_ids=doc_ids)

    if not chunks:
        return {"answer": NOT_FOUND_MESSAGE, "citations": [], "chunks_used": []}

    context = _format_context(chunks)
    user_prompt = (
        f"Context passages from the uploaded research paper(s):\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. If multiple papers are involved, "
        "state clearly which paper each piece of information comes from."
    )

    answer = _call_groq(_SYSTEM_PROMPT, user_prompt)

    # If the model itself declares it can't find the answer, don't attach
    # citations that would misleadingly imply support.
    if NOT_FOUND_MESSAGE.lower() in answer.lower():
        return {"answer": NOT_FOUND_MESSAGE, "citations": [], "chunks_used": []}

    return {
        "answer": answer,
        "citations": format_citations(chunks),
        "chunks_used": chunks,
    }


_SUMMARY_SYSTEM_PROMPT = """You are a research-paper summarization assistant. You must
summarize using ONLY the provided context from a single paper. Organize your summary
under these headings, in this order:

Title
Problem
Objective
Methodology
Dataset
Results
Contributions
Limitations
Conclusion

If the paper's provided context does not contain information for a heading,
write "Not explicitly stated in the provided content" under that heading instead
of inventing information. Do not use outside knowledge.
"""


def summarize_paper(doc_id: str, filename: str) -> Dict:
    """
    Structured, single-paper summary built only from that paper's indexed chunks.
    """
    chunks = vector_store.get_all_chunks_for_doc(doc_id)
    if not chunks:
        raise RAGError(f"No indexed content found for '{filename}'.")

    # Cap total context size sent to the LLM to keep the request reasonable
    # while still covering the whole paper (metadata chunk + representative body chunks).
    max_chunks = 25
    selected = chunks[:max_chunks]
    context = _format_context(selected)

    user_prompt = (
        f"Context passages from the paper '{filename}':\n\n{context}\n\n"
        "Produce the structured summary described in your instructions."
    )

    summary = _call_groq(_SUMMARY_SYSTEM_PROMPT, user_prompt)
    return {
        "summary": summary,
        "citations": format_citations(selected),
    }
