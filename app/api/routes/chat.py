"""POST /chat and GET /chats.

Component 2G · configurable retrieval in 3D · history endpoints in 6A.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import (
    CurrentUserId,
    DbSession,
    EmbedderDep,
    LLMDep,
    RerankerDep,
    SettingsDep,
    retrieval_config,
)
from app.api.rate_limit import rate_limit
from app.api.schemas import (
    ChatConfig,
    ChatRequest,
    ChatResponse,
    CitationOut,
    RetrievedOut,
    Usage,
)
from app.db.repositories.chunks import ChunkHit
from app.rag.generation import BadModelOutputError
from app.rag.llm import LLMError, PromptTooLongError
from app.services.answering import (
    ChatResult,
    DocumentNotFoundError,
    DocumentNotReadyError,
    answer_question,
)

router = APIRouter(tags=["chat"])

QUOTE_CHARS = 300


@router.post("/chat", dependencies=[Depends(rate_limit("chat"))])
def chat(
    request: ChatRequest,
    session: DbSession,
    embedder: EmbedderDep,
    reranker: RerankerDep,
    llm: LLMDep,
    owner_id: CurrentUserId,
    settings: SettingsDep,
) -> ChatResponse:
    config = retrieval_config(settings, request.retrieval)
    try:
        result = answer_question(
            session,
            embedder,
            reranker,
            llm,
            owner_id=owner_id,
            question=request.question,
            document_ids=request.document_ids,
            retrieval=config,
            max_tokens=settings.llm_max_output_tokens,
            prompt_version=settings.answer_prompt_version,
        )
    except DocumentNotFoundError as exc:
        raise error(status.HTTP_404_NOT_FOUND, "document_not_found", exc) from exc
    except DocumentNotReadyError as exc:
        raise error(status.HTTP_409_CONFLICT, "document_not_ready", exc) from exc
    except PromptTooLongError as exc:
        # Too many or too long sources for the model's context window: ask for a smaller top_k.
        raise error(status.HTTP_422_UNPROCESSABLE_CONTENT, "prompt_too_long", exc) from exc
    except LLMError as exc:
        # The model server is down or timed out: a temporary condition, so 503.
        raise error(status.HTTP_503_SERVICE_UNAVAILABLE, "llm_unavailable", exc) from exc
    except BadModelOutputError as exc:
        # The model answered, but with something unusable: a bad upstream reply, so 502.
        raise error(status.HTTP_502_BAD_GATEWAY, "llm_bad_output", exc) from exc
    return to_response(result)


def to_response(result: ChatResult) -> ChatResponse:
    generated = result.generated
    return ChatResponse(
        answer=generated.answer,
        abstained=generated.abstained,
        abstain_reason=generated.abstain_reason,
        citations=[
            CitationOut(marker=c.marker, quote=c.hit.text[:QUOTE_CHARS], **location(c.hit))
            for c in generated.citations
        ],
        retrieved=[retrieved_out(h) for h in result.hits],
        usage=Usage(
            prompt_tokens=sum(c.prompt_tokens for c in generated.llm_calls),
            completion_tokens=sum(c.completion_tokens for c in generated.llm_calls),
            llm_calls=len(generated.llm_calls),
        ),
        timings_ms=result.timings_ms,
        config=ChatConfig(
            retrieval=result.retrieval,
            index_version=result.index_version,
            embedding_model=result.embedding_model,
            reranker_model=result.reranker_model,
            llm_model=result.llm_model,
            prompt_version=generated.prompt_version,
        ),
    )


def retrieved_out(hit: ChunkHit) -> RetrievedOut:
    return RetrievedOut(
        rank=hit.rank,
        score=round(hit.score, 4),
        text=hit.text,
        debug={name: round(value, 4) for name, value in hit.debug.items()},
        **location(hit),
    )


def location(hit: ChunkHit) -> dict:
    return {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "document_title": hit.document_title,
        "section_path": hit.section_path,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
    }


def error(status_code: int, code: str, exc: Exception) -> HTTPException:
    return HTTPException(status_code, {"code": code, "message": str(exc)})
