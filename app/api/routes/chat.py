"""POST /chat and GET /chats.

Component 2G · configurable retrieval in 3D · history endpoints in 6A.
"""

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

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
    ChatList,
    ChatRequest,
    ChatResponse,
    ChatSummary,
    ChatTrace,
    CitationOut,
    RetrievedOut,
    Usage,
)
from app.core.logging import current_request_id, text_fingerprint
from app.core.timing import StageTimer
from app.db.repositories import chats as chats_repo
from app.db.repositories.chunks import ChunkHit
from app.rag.generation import BadModelOutputError
from app.rag.llm import LLMError, PromptTooLongError
from app.rag.pricing import cost_usd
from app.rag.retrieval import RetrievalConfig
from app.services.answering import (
    ChatResult,
    DocumentNotFoundError,
    DocumentNotReadyError,
    answer_question,
)

router = APIRouter(tags=["chat"])
logger = structlog.get_logger(__name__)

QUOTE_CHARS = 300

# Checked in order, so a subclass (PromptTooLongError is an LLMError) comes before its parent.
DOMAIN_ERRORS: list[tuple[type[Exception], int, str]] = [
    (DocumentNotFoundError, status.HTTP_404_NOT_FOUND, "document_not_found"),
    (DocumentNotReadyError, status.HTTP_409_CONFLICT, "document_not_ready"),
    # Too many or too long sources for the model's context window: ask for a smaller top_k.
    (PromptTooLongError, status.HTTP_422_UNPROCESSABLE_CONTENT, "prompt_too_long"),
    # The model server is down or timed out: a temporary condition, so 503.
    (LLMError, status.HTTP_503_SERVICE_UNAVAILABLE, "llm_unavailable"),
    # The model answered, but with something unusable: a bad upstream reply, so 502.
    (BadModelOutputError, status.HTTP_502_BAD_GATEWAY, "llm_bad_output"),
]


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
    """Answer from the caller's documents. Every call is stored, failures included."""
    config = retrieval_config(settings, request.retrieval)
    timer = StageTimer()
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
            timer=timer,
        )
    except Exception as exc:
        status_code, code = next(
            ((s, c) for kind, s, c in DOMAIN_ERRORS if isinstance(exc, kind)),
            (status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error"),
        )
        session.rollback()  # the failure may have left the transaction unusable
        record_failure(session, owner_id, request, config, code, timer.finish(), llm.model)
        if code == "internal_error":
            raise  # the 500 handler logs the stack trace
        raise error(status_code, code, exc) from exc

    response = to_response(result)
    chat_record = chats_repo.create(
        session,
        owner_id,
        request_id=current_request_id(),
        question=request.question,
        document_ids=request.document_ids,
        retrieval_config=config.model_dump(),
        index_version_id=result.index_version_id,
        retrieved=[r.model_dump(mode="json", exclude={"text"}) for r in response.retrieved],
        answer=response.answer,
        abstained=response.abstained,
        abstain_reason=response.abstain_reason,
        citations=[c.model_dump(mode="json") for c in response.citations],
        prompt_version=response.config.prompt_version,
        llm_model=response.config.llm_model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        cost_usd=response.usage.cost_usd,
        timings_ms=response.timings_ms,
    )
    logger.info(
        "chat_answered",
        chat_id=str(chat_record.id),
        question=text_fingerprint(request.question),
        document_filter=len(request.document_ids or []),
        hits=len(result.hits),
        abstained=response.abstained,
        abstain_reason=response.abstain_reason,
        citations=len(response.citations),
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        prompt_version=response.config.prompt_version,
        total_ms=response.timings_ms.get("total"),
    )
    return response.model_copy(update={"id": chat_record.id})


@router.get("/chats")
def list_chats(
    session: DbSession,
    owner_id: CurrentUserId,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChatList:
    chats, total = chats_repo.list_for_owner(session, owner_id, limit=limit, offset=offset)
    return ChatList(
        items=[ChatSummary.model_validate(c) for c in chats],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/chats/{chat_id}")
def get_chat(chat_id: uuid.UUID, session: DbSession, owner_id: CurrentUserId) -> ChatTrace:
    chat_record = chats_repo.get_for_owner(session, owner_id, chat_id)
    if chat_record is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, {"code": "chat_not_found", "message": "Chat not found."}
        )
    return ChatTrace.model_validate(chat_record, from_attributes=True)


def record_failure(
    session,
    owner_id: uuid.UUID,
    request: ChatRequest,
    config: RetrievalConfig,
    error_code: str,
    timings_ms: dict[str, float],
    llm_model: str,
) -> None:
    chats_repo.create(
        session,
        owner_id,
        request_id=current_request_id(),
        question=request.question,
        document_ids=request.document_ids,
        retrieval_config=config.model_dump(),
        llm_model=llm_model,
        timings_ms=timings_ms,
        error_code=error_code,
    )
    logger.info("chat_failed", error_code=error_code, question=text_fingerprint(request.question))


def to_response(result: ChatResult) -> ChatResponse:
    generated = result.generated
    prompt_tokens = sum(c.prompt_tokens for c in generated.llm_calls)
    completion_tokens = sum(c.completion_tokens for c in generated.llm_calls)
    cost = cost_usd(result.llm_model, prompt_tokens, completion_tokens)
    return ChatResponse(
        id=uuid.UUID(int=0),  # replaced by the stored chat's id
        request_id=current_request_id(),
        answer=generated.answer,
        abstained=generated.abstained,
        abstain_reason=generated.abstain_reason,
        citations=[
            CitationOut(marker=c.marker, quote=c.hit.text[:QUOTE_CHARS], **location(c.hit))
            for c in generated.citations
        ],
        retrieved=[retrieved_out(h) for h in result.hits],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            llm_calls=len(generated.llm_calls),
            cost_usd=float(cost) if cost is not None else None,
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
