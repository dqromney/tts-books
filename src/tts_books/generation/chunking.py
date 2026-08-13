"""Split text into TTS-friendly chunks at natural sentence and clause boundaries.

The core of the chunked-generation system: text is split here before the chunk
loop in pipeline.py. On resume, the chunk list is read from state.json — this
function is NOT called again (re-splitting would produce different boundaries
and misalign the completed-set indices).
"""

import re

_SEPARATOR_RE = re.compile(r'^([=\-*~_#])\1{3,}$')


def split_text(text, max_chars=200, split_clauses=True):
    """Split text into TTS-friendly chunks at natural boundaries."""
    sent_end = r'(?<=[.!?])[\""’]?\s+'
    clause_end = r'|(?<=[;:])\s+'
    boundary = re.compile(sent_end + (clause_end if split_clauses else ''))

    sentences = []
    for para in re.split(r'\n\s*\n', text.strip()):
        para = para.strip()
        if not para:
            continue
        lines = [ln for ln in para.splitlines() if not _SEPARATOR_RE.match(ln.strip())]
        para = '\n'.join(lines).strip()
        if not para:
            continue
        para = re.sub(r'(?<!\n)\n(?!\n)', ' ', para)
        para = re.sub(r' {2,}', ' ', para)
        for s in boundary.split(para):
            s = s.strip()
            if s:
                sentences.append(s)

    def _opener(s):
        """Return the first-2-word tuple of a sentence (used for anaphora guard)."""
        ws = s.split()
        return tuple(ws[:2]) if len(ws) >= 2 else (tuple(ws) if ws else ())

    chunks = []
    current = ""
    current_openers = []   # first-2-word tuples of sentences already in current chunk

    for s in sentences:
        opener = _opener(s)
        if not current:
            current = s
            current_openers = [opener]
        elif len(current) + 1 + len(s) <= max_chars:
            # Anaphora guard: avoid 3+ consecutive sentences with the same
            # first-2-word opener in one chunk — the TTS tends to loop on them.
            if opener and current_openers.count(opener) >= 2:
                chunks.append(current)
                current = s
                current_openers = [opener]
            else:
                current = current + " " + s
                current_openers.append(opener)
        else:
            chunks.append(current)
            current = s
            current_openers = [opener]
        while len(current) > max_chars:
            semi_pos = current.rfind(';', 0, max_chars)
            comma_pos = current.rfind(',', 0, max_chars)
            space_pos = current.rfind(' ', 0, max_chars)
            split_at = max(semi_pos, comma_pos, space_pos)
            if split_at <= 0:
                split_at = max_chars
            chunks.append(current[:split_at].strip())
            rest = current[split_at:].strip()
            # Capitalize continuations split at a semicolon (Appendix list items
            # starting mid-sentence confuse the TTS without a sentence-start signal).
            if split_at == semi_pos and rest and rest[0].islower():
                rest = rest[0].upper() + rest[1:]
            current = rest
            current_openers = [_opener(current)]
    if current:
        chunks.append(current)
    return chunks if chunks else [text]
