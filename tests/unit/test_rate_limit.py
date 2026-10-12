import fakeredis
from redis import RedisError

from app.api.rate_limit import RateLimiter


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def limiter(clock: Clock) -> RateLimiter:
    return RateLimiter(fakeredis.FakeRedis(), clock=clock)


def test_requests_within_the_limit_are_allowed_and_counted_down():
    rl = limiter(Clock(1_000_020.0))

    decisions = [rl.hit("chat:user", limit=3) for _ in range(3)]

    assert [d.allowed for d in decisions] == [True, True, True]
    assert [d.remaining for d in decisions] == [2, 1, 0]


def test_the_request_over_the_limit_is_refused_until_the_window_ends():
    clock = Clock(1_000_040.0)  # 1_000_020 is divisible by 60: 20 s into a window
    rl = limiter(clock)
    for _ in range(3):
        rl.hit("chat:user", limit=3)

    refused = rl.hit("chat:user", limit=3)

    assert not refused.allowed
    assert refused.retry_after == 40


def test_a_new_window_starts_a_new_count():
    clock = Clock(1_000_020.0)
    rl = limiter(clock)
    for _ in range(4):
        rl.hit("chat:user", limit=3)

    clock.now += 60

    assert rl.hit("chat:user", limit=3).allowed


def test_keys_have_separate_budgets():
    rl = limiter(Clock(1_000_020.0))
    for _ in range(3):
        rl.hit("chat:alice", limit=3)

    assert not rl.hit("chat:alice", limit=3).allowed
    assert rl.hit("chat:bob", limit=3).allowed
    assert rl.hit("retrieve:alice", limit=3).allowed


def test_counters_expire_so_redis_does_not_fill_up():
    redis = fakeredis.FakeRedis()
    RateLimiter(redis, clock=Clock(1_000_020.0)).hit("chat:user", limit=3)

    (key,) = redis.keys("ratelimit:*")

    assert 0 < redis.ttl(key) <= 61


def test_redis_outage_fails_open():
    class DownRedis:
        def pipeline(self, transaction=True):
            raise RedisError("connection refused")

    decision = RateLimiter(DownRedis(), clock=Clock(1_000_020.0)).hit("chat:user", limit=3)

    assert decision.allowed
