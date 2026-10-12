from app.core.security import (
    KEY_PREFIX,
    display_prefix,
    generate_api_key,
    hash_api_key,
    looks_like_api_key,
)


def test_keys_have_the_prefix_and_32_random_bytes():
    key = generate_api_key()

    assert key.startswith(KEY_PREFIX)
    assert len(key) == len(KEY_PREFIX) + 43  # 32 bytes in url-safe base64
    assert looks_like_api_key(key)


def test_keys_are_unique():
    assert len({generate_api_key() for _ in range(1000)}) == 1000


def test_hash_is_sha256_hex_and_deterministic():
    key = generate_api_key()

    assert hash_api_key(key) == hash_api_key(key)
    assert len(hash_api_key(key)) == 64
    assert hash_api_key(key) != hash_api_key(key + "x")


def test_display_prefix_reveals_only_the_start():
    key = generate_api_key()

    assert display_prefix(key) == key[:12]


def test_format_check_rejects_other_tokens():
    assert not looks_like_api_key("")
    assert not looks_like_api_key("sk-somebody-elses-token")
    assert not looks_like_api_key(KEY_PREFIX + "a" * 200)
