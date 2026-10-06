"""The same hits as Anthropic Messages API search_result blocks."""

from policy_retrieval import NO_RESULTS, Hit, provenance


def to_search_results(hits: list[Hit]) -> list[dict]:
    """Content for a tool_result. Every block must be a search_result, so an
    empty search returns a plain text block instead."""
    if not hits:
        return [{"type": "text", "text": NO_RESULTS}]
    return [{"type": "search_result",
             "source": f"policy://{h.chunk.chunk_id}",
             "title": provenance(h.chunk.doc),
             "content": [{"type": "text", "text": h.chunk.text}],
             "citations": {"enabled": True}}
            for h in hits]
