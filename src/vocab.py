from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Tuple

_VOCAB_CACHE: Dict[int, Dict[int, str]] = {}
_BUCKET_CACHE: Dict[int, Dict[str, List[Tuple[int, str]]]] = {}
_DEBUG = os.environ.get("DECODER_DEBUG") == "1"


def _decode_token_piece(model: Any, token_id: int) -> str:
    piece = model.decode([token_id])
    if not isinstance(piece, str):
        raise ValueError(
            f"Decoded piece for token {token_id} is not a string"
        )
    return piece


def get_vocab_map(model: Any, vocab_size: int) -> Dict[int, str]:
    cache_key = id(model)
    cached = _VOCAB_CACHE.get(cache_key)
    if cached is not None:
        return cached

    start = time.monotonic()
    vocab: Dict[int, str] = {}
    for token_id in range(vocab_size):
        try:
            vocab[token_id] = _decode_token_piece(model, token_id)
        except Exception:
            continue

    if _DEBUG:
        elapsed = time.monotonic() - start
        print(
            f"[vocab] built vocab map: {len(vocab)}/{vocab_size} "
            f"tokens in {elapsed:.2f}s",
            file=sys.stderr,
        )

    _VOCAB_CACHE[cache_key] = vocab
    return vocab


def get_leading_char_buckets(
    model: Any, vocab_map: Dict[int, str]
) -> Dict[str, List[Tuple[int, str]]]:
    cache_key = id(model)
    cached = _BUCKET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    start = time.monotonic()
    buckets: Dict[str, List[Tuple[int, str]]] = {}
    for token_id, piece in vocab_map.items():
        if not piece:
            continue
        key = piece[0]
        buckets.setdefault(key, []).append((token_id, piece))

    if _DEBUG:
        elapsed = time.monotonic() - start
        print(
            f"[vocab] built {len(buckets)} leading-char buckets "
            f"in {elapsed:.2f}s",
            file=sys.stderr,
        )

    _BUCKET_CACHE[cache_key] = buckets
    return buckets
