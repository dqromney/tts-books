"""Garbled-chunk detection — 10-heuristic classifier for TTS output quality.

Entry point: is_chunk_garbled(wav_path, text, whisper_handle, logger)
Returns a reason string if garbled, or None if the chunk sounds clean.

All ten checks run in order from cheapest (duration heuristic) to most
expensive (STT-based n-gram analysis). The function short-circuits on the
first positive (garbled) signal.

Heuristic summary:
  1. Duration — too short (cutoff) or too long (loop)
  2. Word overlap — fraction of expected words in STT transcript < 45%
  3. Word-count ratio — spoken / expected < 50%
  4. Unrecognized tail — last word ends > 2.5 s before audio ends
  5. Tail-word mismatch — last 6 words < 35% match expected set
  6. Word stutter — consecutive identical tokens not in source
  7. Final content word absent — last unhyphenated ≥5-char word not transcribed
  8. Phrase repetition loop — 5-gram appears twice but not in source
  9. Anaphoric over-repetition — 5-gram appears more times than in source
 10. Single-source phrase spoken twice — 6-gram once in source, ≥2x in transcript
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import torchaudio

if TYPE_CHECKING:
    from tts_books.gui import Logger
    from tts_books.quality.whisper_backend import WhisperHandle

_TOKEN_RE = re.compile(r'[^a-z0-9]')


def _tokenize(w: str) -> str:
    return _TOKEN_RE.sub('', w.lower())


def is_chunk_garbled(
    wav_path: str,
    text: str,
    whisper_handle: WhisperHandle,
    logger: Logger,
) -> str | None:
    """Return a reason string if garbled, or None if the chunk sounds clean."""
    try:
        info = torchaudio.info(wav_path)
        duration = info.num_frames / info.sample_rate
        n_chars = len(text.strip())
        if n_chars < 10:
            return None
        if duration < (n_chars / 25.0):
            return "too short"
        if duration > (n_chars / 3.5):
            return "too long"
    except Exception:
        return None

    n_words = len(text.split())
    if n_words < 5:
        return None

    # Build expected word set (checks 2, 4, 5, 7).
    # Exclude hyphenated words — pronunciation substitutions like "ay-bish" are
    # transcribed by Whisper as their audio sound, not their spelling, so they'd
    # artificially deflate overlap scores.
    expected_set = {_tokenize(w) for w in text.split() if len(w) >= 3 and '-' not in w}

    try:
        if not whisper_handle.is_loaded and not whisper_handle.is_unavailable:
            logger.info("  Loading faster-whisper tiny.en for garble detection…")
        wm = whisper_handle.load()
        if wm is None:
            return None

        segments, _ = wm.transcribe(wav_path, language="en", beam_size=1, word_timestamps=True)
        segments = list(segments)
        if not segments:
            return "STT: no transcription"

        transcribed_text = ' '.join(s.text for s in segments)

        # Check 2: word overlap
        transcribed_set = {_tokenize(w) for w in transcribed_text.split() if len(w) >= 3}
        if expected_set:
            overlap = len(transcribed_set & expected_set) / len(expected_set)
            if overlap < 0.45:
                return (f"STT: {overlap:.0%} word overlap "
                        f"({len(transcribed_set & expected_set)}/{len(expected_set)} matched)")

        # Check 3: word-count ratio
        spoken_words = sum(len(s.text.split()) for s in segments)
        if spoken_words / n_words < 0.5:
            return f"STT: {spoken_words}/{n_words} words ({spoken_words/n_words:.0%})"

        all_words = [w for s in segments for w in (s.words or [])]

        # Check 4: unrecognized tail
        if all_words:
            last_word_end = all_words[-1].end
            unrecognized_tail = duration - last_word_end
            if unrecognized_tail > 2.5:
                return (f"STT: {unrecognized_tail:.1f}s unrecognized tail"
                        f" (whisper stopped at {last_word_end:.1f}s/{duration:.1f}s)")

        # Check 5: garbled tail words
        if len(all_words) >= 6:
            end_words = all_words[-6:]
            end_tokens = [_tokenize(w.word) for w in end_words if len(w.word.strip()) >= 3]
            if len(end_tokens) >= 2:
                match_frac = sum(1 for t in end_tokens if t in expected_set) / len(end_tokens)
                if match_frac < 0.35:
                    return f"STT: tail garbled ({match_frac:.0%} end words recognized)"

        # Check 6: word stutter
        word_tokens_all = [_tokenize(w.word) for w in all_words]
        exp_tokens_seq = [_tokenize(w) for w in text.split()]
        expected_consec = {
            exp_tokens_seq[i] for i in range(1, len(exp_tokens_seq))
            if exp_tokens_seq[i] and exp_tokens_seq[i] == exp_tokens_seq[i - 1]
        }
        stutter_pairs = [
            word_tokens_all[i] for i in range(1, len(word_tokens_all))
            if (word_tokens_all[i]
                and word_tokens_all[i] == word_tokens_all[i - 1]
                and word_tokens_all[i] not in expected_consec
                and not word_tokens_all[i].isdigit())
        ]
        if stutter_pairs:
            sample = stutter_pairs[0]
            extra = f", {len(stutter_pairs)} pairs" if len(stutter_pairs) > 1 else ""
            return f"STT: word stutter ('{sample}' repeated{extra})"

        # Check 7: final content word absent
        last_content_raw = next(
            (w for w in reversed(text.split()) if '-' not in w and re.match(r'[a-zA-Z]{5}', w)),
            None,
        )
        if last_content_raw:
            last_content = _tokenize(last_content_raw)
            vowel_count = sum(1 for c in last_content if c in 'aeiou')
            whisper_knows = len(last_content) <= 3 or (vowel_count / len(last_content)) >= 0.2
            if whisper_knows:
                transcribed_tokens = {_tokenize(w) for w in transcribed_text.split()}
                if last_content not in transcribed_tokens:
                    return f"STT: final content word '{last_content}' not spoken"

        # Check 8: phrase repetition loop (5-gram)
        _N = 5
        exp_toks = [_tokenize(w) for w in text.split() if _tokenize(w)]
        if len(word_tokens_all) >= _N * 2:
            exp_ngrams = {
                tuple(exp_toks[i:i + _N]) for i in range(len(exp_toks) - _N + 1)
            }
            seen_ngrams: set = set()
            for i in range(len(word_tokens_all) - _N + 1):
                gram = tuple(t for t in word_tokens_all[i:i + _N] if t)
                if len(gram) < _N:
                    continue
                if gram in seen_ngrams and gram not in exp_ngrams:
                    preview = ' '.join(gram[:4]) + '…'
                    return f"STT: phrase repetition loop ('{preview}')"
                seen_ngrams.add(gram)

        # Check 9: anaphoric over-repetition (5-gram)
        if len(word_tokens_all) >= _N * 2:
            exp_ngram_counts = Counter(
                tuple(exp_toks[i:i + _N]) for i in range(len(exp_toks) - _N + 1)
            )
            trx_ngram_counts = Counter(
                tuple(t for t in word_tokens_all[i:i + _N] if t)
                for i in range(len(word_tokens_all) - _N + 1)
            )
            for gram, trx_count in trx_ngram_counts.items():
                src_count = exp_ngram_counts.get(gram, 0)
                if src_count >= 2 and trx_count > src_count:
                    preview = ' '.join(gram[:4]) + '…'
                    return (f"STT: anaphoric loop ('{preview}' "
                            f"{trx_count}x spoken, {src_count}x in source)")

        # Check 10: single-source phrase spoken twice (6-gram)
        _N6 = 6
        if len(word_tokens_all) >= _N6 * 2 and len(exp_toks) >= _N6:
            exp_ngram_counts6 = Counter(
                tuple(exp_toks[i:i + _N6]) for i in range(len(exp_toks) - _N6 + 1)
            )
            trx_ngram_counts6 = Counter(
                tuple(t for t in word_tokens_all[i:i + _N6] if t)
                for i in range(len(word_tokens_all) - _N6 + 1)
            )
            for gram, trx_count in trx_ngram_counts6.items():
                if len(gram) < _N6:
                    continue
                src_count = exp_ngram_counts6.get(gram, 0)
                if src_count == 1 and trx_count >= 2:
                    preview = ' '.join(gram[:5]) + '…'
                    return (f"STT: phrase spoken twice ('{preview}' "
                            f"{trx_count}x spoken, 1x in source)")

    except Exception:
        pass
    return None
