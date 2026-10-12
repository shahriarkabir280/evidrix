"""Per-user request limits in Redis: a fixed window per route.

Component 5E · Plan: Part 5, step 6.

Fixed window: the minute 12:00:00-12:00:59 has one counter, 12:01 the next. It is one INCR per
request and easy to reason about; the known weakness is a burst at a window edge (up to twice
the limit within a few seconds around 12:01:00). A sliding window fixes that at the cost of
more Redis work, which these limits don't need.

Limits are per user, not per key: a user holding several keys doesn't get several budgets.
Unauthenticated requests never reach the limiter (the 401 comes first); flooding with bad keys
is for a limit by IP at the reverse proxy (Part 7).
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Literal

import structlog
from fastapi import Depends, HTTPException, Response, status
from redis import Redis, RedisError

from app.api.deps import CurrentUserId, SettingsDep
from app.core.config import get_settings

logger = structlog.get_logger(__name__)

Route = Literal["chat", "retrieve", "upload"]
WINDOW_SECONDS = 60


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # seconds until the window resets


class RateLimiter:
    def __init__(self, redis: Redis, clock: Callable[[], float] = time.time) -> None:
        self._redis = redis
        self._clock = clock

    def hit(self, key: str, limit: int, window_seconds: int = WINDOW_SECONDS) -> Decision:
        """Count one request against `key`; Redis errors fail open (see below)."""
        now = self._clock()
        window = int(now // window_seconds)
        retry_after = max(1, math.ceil((window + 1) * window_seconds - now))
        redis_key = f"ratelimit:{key}:{window}"
        try:
            # MULTI/EXEC: the counter and its expiry are set together, or not at all.
            pipe = self._redis.pipeline(transaction=True)
            pipe.incr(redis_key)
            pipe.expire(redis_key, window_seconds + 1)
            count, _ = pipe.execute()
        except RedisError:
            # Fail open: a Redis outage shouldn't take reading and chatting down with it.
            # The trade-off is that limits aren't enforced until Redis is back.
            logger.warning("rate_limiter_unavailable", exc_info=True)
            return Decision(True, limit, limit, retry_after)
        return Decision(count <= limit, limit, max(0, limit - count), retry_after)


@lru_cache
def get_rate_limiter() -> RateLimiter:
    # Short timeouts: the limiter is on every request's path, so a slow Redis must not make
    # every request slow. After 0.5 s the request goes through unlimited instead.
    redis = Redis.from_url(get_settings().redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)
    return RateLimiter(redis)


RateLimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]


def rate_limit(route: Route) -> Callable[..., None]:
    """A route dependency: `dependencies=[Depends(rate_limit("chat"))]`."""

    def check(
        owner_id: CurrentUserId, limiter: RateLimiterDep, settings: SettingsDep, response: Response
    ) -> None:
        limit = getattr(settings, f"rate_limit_{route}_per_minute")
        decision = limiter.hit(f"{route}:{owner_id}", limit)
        headers = {
            "X-RateLimit-Limit": str(decision.limit),
            "X-RateLimit-Remaining": str(decision.remaining),
        }
        if not decision.allowed:
            logger.info("rate_limited", route=route, limit=limit)
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                {
                    "code": "rate_limited",
                    "message": f"Limit of {limit} {route} requests per minute reached.",
                },
                headers={**headers, "Retry-After": str(decision.retry_after)},
            )
        response.headers.update(headers)

    return check
