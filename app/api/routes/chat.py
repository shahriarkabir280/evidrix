"""POST /chat and GET /chats.

Component 2G · history endpoints in 6A.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUserId, DbSession, EmbedderDep, LLMDep, SettingsDep
from app.api.schemas import (
    ChatConfig,
    ChatRequest,
    ChatResponse,
    CitationOut,
    RetrievedOut,
    Usage,
)
from app.rag.generation import BadModelOutputError
from app.rag.llm import LLMError
from app.services.answering import (
    ChatResult,
    DocumentNotFoundError,
    DocumentNotReadyError,
    answer_question,
)

router = APIRouter(tags=["chat"])

QUOTE_CHARS = 300


@router.post("/chat")
def chat(
    request: ChatRequest,
    session: DbSession,
    embedder: EmbedderDep,
    llm: LLMDep,
    owner_id: CurrentUserId,
    settings: SettingsDep,
) -> ChatResponse:
    try:
        result = answer_question(
            session,
            embedder,
            llm,
            owner_id=owner_id,
            question=request.question,
            document_ids=request.document_ids,
            top_k=settings.retrieval_top_k,
            max_tokens=settings.llm_max_output_tokens,
            min_score=settings.abstain_min_score,
        )
    except DocumentNotFoundError as exc:
        raise _error(status.HTTP_404_NOT_FOUND, "document_not_found", exc) from exc
    except DocumentNotReadyError as exc:
        raise _error(status.HTTP_409_CONFLICT, "document_not_ready", exc) from exc
    except LLMError as exc:
        # The model server is down or timed out: a temporary condition, so 503.
        raise _error(status.HTTP_503_SERVICE_UNAVAILABLE, "llm_unavailable", exc) from exc
    except BadModelOutputError as exc:
        # The model answered, but with something unusable: a bad upstream reply, so 502.
        raise _error(status.HTTP_502_BAD_GATEWAY, "llm_bad_output", exc) from exc
    return to_response(result)


def to_response(result: ChatResult) -> ChatResponse:
    generated = result.generated
    return ChatResponse(
        answer=generated.answer,
        abstained=generated.abstained,
        abstain_reason=generated.abstain_reason,
        citations=[
            CitationOut(
                marker=c.marker,
                quote=c.hit.text[:QUOTE_CHARS],
                **_location(c.hit),
            )
            for c in generated.citations
        ],
        retrieved=[
            RetrievedOut(rank=h.rank, score=round(h.score, 4), text=h.text, **_location(h))
            for h in result.hits
        ],
        usage=Usage(
            prompt_tokens=sum(c.prompt_tokens for c in generated.llm_calls),
            completion_tokens=sum(c.completion_tokens for c in generated.llm_calls),
            llm_calls=len(generated.llm_calls),
        ),
        timings_ms=result.timings_ms,
        config=ChatConfig(
            top_k=result.top_k,
            index_version=result.index_version,
            embedding_model=result.embedding_model,
            llm_model=result.llm_model,
            prompt_version=generated.prompt_version,
        ),
    )


def _location(hit) -> dict:
    return {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "document_title": hit.document_title,
        "section_path": hit.section_path,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
    }


def _error(status_code: int, code: str, exc: Exception) -> HTTPException:
    return HTTPException(status_code, {"code": code, "message": str(exc)})
