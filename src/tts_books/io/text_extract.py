"""Load and pre-process text files for TTS.

Supports plain text, PDF (via PyMuPDF), EPUB (via ebooklib), and Markdown.
Markdown is converted to clean plain text before chunking.
"""

import re


def load_text_file(path):
    if path.lower().endswith('.pdf'):
        import fitz
        doc = fitz.open(path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    elif path.lower().endswith('.epub'):
        import ebooklib
        from bs4 import BeautifulSoup
        from ebooklib import epub
        book = epub.read_epub(path)
        texts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            texts.append(soup.get_text())
        return '\n\n'.join(texts)
    elif path.lower().endswith('.md'):
        with open(path, encoding='utf-8') as f:
            raw = f.read()
        return _strip_markdown(raw)
    else:
        with open(path, encoding='utf-8') as f:
            return f.read()


def _strip_markdown(text):
    """Convert markdown to clean plain text for TTS."""
    lines = text.split('\n')
    out = []
    in_code = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue

        if re.match(r'^[-*_]{3,}\s*$', stripped):
            continue

        heading = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading:
            if len(heading.group(1)) == 1:
                continue
            out.append(heading.group(2))
            out.append('')
            continue

        if '|' in stripped and not stripped.startswith('>'):
            if re.match(r'^[\|\s\-:]+$', stripped):
                continue
            cells = [c.strip() for c in stripped.split('|') if c.strip()]
            if cells:
                out.append('. '.join(cells) + '.')
            continue

        if stripped.startswith('>'):
            content = re.sub(r'^>\s*', '', stripped).strip()
            if re.match(r'^\*{0,2}(Chapter status|Word count|Thread|Biblical anchor|Next)\b', content, re.IGNORECASE):
                continue
            out.append(content)
            continue

        li = re.match(r'^[\*\-\+]\s+(.+)$', stripped)
        num_li = re.match(r'^\d+\.\s+(.+)$', stripped)
        if li:
            out.append(_clean_inline(li.group(1)))
            continue
        if num_li:
            out.append(_clean_inline(num_li.group(1)))
            continue

        line = _clean_inline(line)

        if stripped == '':
            out.append('')
        elif line.strip():
            out.append(line.strip())

    return '\n'.join(out)


def _clean_inline(line):
    """Strip inline markdown formatting from a single line."""
    line = re.sub(r'!\[.*?\]\(.*?\)', '', line)
    line = re.sub(r'\[(.+?)\]\(.*?\)', r'\1', line)
    line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
    line = re.sub(r'__(.+?)__', r'\1', line)
    line = re.sub(r'\*(.+?)\*', r'\1', line)
    line = re.sub(r'_(.+?)_', r'\1', line)
    line = re.sub(r'`(.+?)`', r'\1', line)
    line = re.sub(r'<[^>]+>', '', line)
    line = re.sub(r' {2,}', ' ', line)
    return line.strip()
