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
from app.rag.conversation import MAX_TURNS, Turn
from app.rag.generation import BadModelOutputError
from app.rag.llm import LLMBusyError, LLMError, PromptTooLongError
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
LLM_BUSY_RETRY_AFTER_SECONDS = 30  # about two generations at the measured p95

# Checked in order, so a subclass (PromptTooLongError is an LLMError) comes before its parent.
DOMAIN_ERRORS: list[tuple[type[Exception], int, str]] = [
    (DocumentNotFoundError, status.HTTP_404_NOT_FOUND, "document_not_found"),
    (DocumentNotReadyError, status.HTTP_409_CONFLICT, "document_not_ready"),
    # Too many or too long sources for the model's context window: ask for a smaller top_k.
    (PromptTooLongError, status.HTTP_422_UNPROCESSABLE_CONTENT, "prompt_too_long"),
    # Every model slot stayed busy: overloaded, so 503 with a hint when to retry.
    (LLMBusyError, status.HTTP_503_SERVICE_UNAVAILABLE, "llm_busy"),
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
    if not settings.chat_enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {
                "code": "chat_disabled",
                "message": "This deployment is retrieval-only (no LLM): use POST /retrieve.",
            },
        )
    config = retrieval_config(settings, request.retrieval)
    conversation_id, history = load_conversation(session, owner_id, request.conversation_id)
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
            history=history,
            # The threshold was calibrated on questions about content, searched across all of a
            # user's documents. A user who picked the documents has said where the answer is,
            # and a question about the document itself ("give me the abstract") shares no words
            # with the passage, so the reranker scores it far below the threshold (-9.8 vs -2.84)
            # even when the right passage is first. The model reads the passages instead, and
            # still abstains, with citation checks, when they don't answer the question.
            score_gate=not request.document_ids,
        )
    except Exception as exc:
        status_code, code = next(
            ((s, c) for kind, s, c in DOMAIN_ERRORS if isinstance(exc, kind)),
            (status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error"),
        )
        session.rollback()  # the failure may have left the transaction unusable
        record_failure(
            session, owner_id, request, config, code, timer.finish(), llm.model, conversation_id
        )
        if code == "internal_error":
            raise  # the 500 handler logs the stack trace
        busy = {"Retry-After": str(LLM_BUSY_RETRY_AFTER_SECONDS)} if code == "llm_busy" else None
        raise error(status_code, code, exc, headers=busy) from exc

    response = to_response(result).model_copy(update={"conversation_id": conversation_id})
    chat_record = chats_repo.create(
        session,
        owner_id,
        conversation_id=conversation_id,
        request_id=current_request_id(),
        question=request.question,
        search_query=result.search_query,
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
        follow_up=bool(history),
        rewritten=result.search_query is not None,
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


def load_conversation(
    session, owner_id: uuid.UUID, conversation_id: uuid.UUID | None
) -> tuple[uuid.UUID, list[Turn]]:
    """The conversation to add to, and its last turns. A new id when none was given.

    An id with no chats of this user's is refused like any other resource that isn't theirs:
    404, whether it belongs to someone else or never existed.
    """
    if conversation_id is None:
        return uuid.uuid4(), []
    turns = chats_repo.conversation_turns(session, owner_id, conversation_id, MAX_TURNS)
    if not turns:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            {"code": "conversation_not_found", "message": "Conversation not found."},
        )
    # The standalone form of earlier follow-ups, so that "and the third?" resolves through them.
    history = [
        Turn(question=t.search_query or t.question, answer=None if t.abstained else t.answer)
        for t in turns
    ]
    return conversation_id, history


@router.get("/chats")
def list_chats(
    session: DbSession,
    owner_id: CurrentUserId,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    conversation_id: uuid.UUID | None = None,
) -> ChatList:
    """Newest first. With conversation_id, only that conversation's chats."""
    chats, total = chats_repo.list_for_owner(
        session, owner_id, limit=limit, offset=offset, conversation_id=conversation_id
    )
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
    conversation_id: uuid.UUID,
) -> None:
    chats_repo.create(
        session,
        owner_id,
        conversation_id=conversation_id,
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
    calls = [*([result.rewrite_call] if result.rewrite_call else []), *generated.llm_calls]
    prompt_tokens = sum(c.prompt_tokens for c in calls)
    completion_tokens = sum(c.completion_tokens for c in calls)
    cost = cost_usd(result.llm_model, prompt_tokens, completion_tokens)
    return ChatResponse(
        id=uuid.UUID(int=0),  # replaced by the stored chat's id
        request_id=current_request_id(),
        conversation_id=uuid.UUID(int=0),  # set by the caller
        search_query=result.search_query,
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
            llm_calls=len(calls),
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
        "document_title": hit.document_display_title or hit.document_title,
        "section_path": hit.section_path,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
    }


def error(
    status_code: int, code: str, exc: Exception, headers: dict[str, str] | None = None
) -> HTTPException:
    return HTTPException(status_code, {"code": code, "message": str(exc)}, headers=headers)
