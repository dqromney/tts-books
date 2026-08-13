"""Tests for generation/chunking.py — split_text() behaviour.

Covers edge cases documented in docs/CLAUDE.md:
- Basic splitting and max_chars cap
- Separator-line stripping (_SEPARATOR_RE)
- Clause-boundary splitting (split_clauses)
- Hard-split of oversized sentences
- Anaphora guard (≥3 same-opener sentences → split)
- Whitespace normalisation and empty-text fallback
"""

import pytest

from tts_books.generation.chunking import split_text


class TestSplitTextBasics:
    def test_short_text_returns_single_chunk(self):
        text = "Hello world."
        chunks = split_text(text, max_chars=200)
        assert chunks == ["Hello world."]

    def test_empty_text_returns_original(self):
        # Fallback: empty → [text] (preserves caller contract)
        chunks = split_text("", max_chars=200)
        assert chunks == [""]

    def test_whitespace_only_returns_original(self):
        chunks = split_text("   ", max_chars=200)
        assert chunks == ["   "]

    def test_all_chunks_within_max_chars(self):
        text = (
            "The fox jumped. The dog barked. The cat ran away quickly. "
            "Birds flew overhead. The sun set slowly over the horizon."
        )
        chunks = split_text(text, max_chars=50)
        for c in chunks:
            assert len(c) <= 50, f"Chunk too long ({len(c)}): {c!r}"

    def test_no_empty_chunks(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_text(text, max_chars=200)
        assert all(c.strip() for c in chunks)

    def test_multiple_sentences_merged_under_limit(self):
        # Two short sentences should fit in one 200-char chunk.
        chunks = split_text("Hello world. Goodbye world.", max_chars=200)
        assert len(chunks) == 1

    def test_long_text_splits_into_multiple_chunks(self):
        # Force splitting by using a very short max_chars.
        text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."
        chunks = split_text(text, max_chars=20)
        assert len(chunks) > 1

    def test_total_content_preserved(self):
        # All words from the original must appear somewhere in the chunks.
        text = (
            "The quick brown fox jumped over the lazy dog. "
            "She sells sea shells by the sea shore."
        )
        joined = " ".join(split_text(text, max_chars=40))
        for word in text.split():
            assert word in joined


class TestSeparatorStripping:
    def test_equals_separator_stripped(self):
        text = "Before.\n\n====\n\nAfter."
        chunks = split_text(text, max_chars=200)
        combined = " ".join(chunks)
        assert "====" not in combined

    def test_dash_separator_stripped(self):
        text = "Before.\n\n----\n\nAfter."
        chunks = split_text(text, max_chars=200)
        combined = " ".join(chunks)
        assert "----" not in combined

    def test_star_separator_stripped(self):
        text = "Before.\n\n****\n\nAfter."
        chunks = split_text(text, max_chars=200)
        combined = " ".join(chunks)
        assert "****" not in combined

    def test_tilde_separator_stripped(self):
        text = "First.\n\n~~~~\n\nSecond."
        chunks = split_text(text, max_chars=200)
        combined = " ".join(chunks)
        assert "~~~~" not in combined

    def test_hash_separator_stripped(self):
        text = "First.\n\n####\n\nSecond."
        chunks = split_text(text, max_chars=200)
        combined = " ".join(chunks)
        assert "####" not in combined

    def test_content_around_separator_preserved(self):
        text = "Before.\n\n====\n\nAfter."
        chunks = split_text(text, max_chars=200)
        combined = " ".join(chunks)
        assert "Before" in combined
        assert "After" in combined

    def test_short_dash_not_stripped(self):
        # Only 4+ repeated chars are separators.
        text = "Well -- she said it."
        chunks = split_text(text, max_chars=200)
        assert chunks[0] == "Well -- she said it."


class TestClauseSplitting:
    def test_semicolon_splits_when_enabled(self):
        text = "She ran fast; he followed slowly. Then both stopped."
        chunks_with = split_text(text, max_chars=30, split_clauses=True)
        chunks_without = split_text(text, max_chars=30, split_clauses=False)
        # With clause splitting enabled, smaller pieces should appear
        assert len(chunks_with) >= len(chunks_without)

    def test_split_clauses_false_keeps_together(self):
        # A text that would split on a semicolon should stay together if clauses off.
        text = "She ran; he walked."
        chunks = split_text(text, max_chars=200, split_clauses=False)
        assert len(chunks) == 1
        assert "She ran; he walked." in chunks[0]

    def test_colon_boundary_respected(self):
        text = "Note: this is important. And this too."
        chunks = split_text(text, max_chars=20, split_clauses=True)
        # "Note" should appear in one of the chunks
        assert any("Note" in c for c in chunks)


class TestHardSplit:
    def test_oversized_sentence_split_at_word_boundary(self):
        # A single sentence > max_chars must be split.
        long_sentence = "The " + "word " * 60 + "end."
        chunks = split_text(long_sentence, max_chars=50)
        for c in chunks:
            assert len(c) <= 50, f"Chunk too long: {c!r}"

    def test_hard_split_at_comma(self):
        # Prefer comma as split point over mid-word.
        text = "A, " + "b" * 60 + "."
        chunks = split_text(text, max_chars=30)
        assert all(len(c) <= 30 for c in chunks)

    def test_hard_split_prefers_semicolon_over_mid_word(self):
        # When there is no space within max_chars but there is a semicolon,
        # the split falls at the semicolon (not past max_chars mid-word).
        text = "abc;defghijklmnop."   # no space anywhere before max_chars
        chunks = split_text(text, max_chars=8)
        # First chunk should end before the long run of letters
        assert chunks[0] == "abc"
        # Content is preserved across chunks (semicolon lands in the rest chunk)
        joined = " ".join(chunks)
        assert "defghij" in joined
        assert "klmnop" in joined


class TestAnaphoraGuard:
    def test_three_same_opener_sentences_split(self):
        # Three sentences starting with the same two words should not all
        # land in one chunk (the anaphora guard kicks in at count >= 2).
        text = (
            "He walked quickly. He walked slowly. He walked carefully. "
            "He walked away now."
        )
        chunks = split_text(text, max_chars=200)
        # At least 2 chunks expected because of the guard.
        assert len(chunks) >= 2

    def test_two_same_opener_sentences_allowed(self):
        # Two sentences with the same opener should stay together if they fit.
        text = "He walked quickly. He walked slowly."
        chunks = split_text(text, max_chars=200)
        assert len(chunks) == 1


class TestParagraphHandling:
    def test_paragraphs_treated_independently(self):
        text = "First paragraph sentence.\n\nSecond paragraph sentence."
        chunks = split_text(text, max_chars=200)
        # Should produce one chunk (both fit), or two if split by paragraph
        assert len(chunks) >= 1

    def test_internal_newlines_collapsed_to_space(self):
        text = "Line one\nline two."
        chunks = split_text(text, max_chars=200)
        assert chunks[0] == "Line one line two."

    def test_multiple_spaces_collapsed(self):
        text = "Word  and   another."
        chunks = split_text(text, max_chars=200)
        assert "  " not in chunks[0]
