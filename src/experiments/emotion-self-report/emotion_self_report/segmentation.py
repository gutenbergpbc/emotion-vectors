"""Exact-span sentence segmentation, promoted from the coding baseline.

Source: experiments/gpt_oss_coding_early_warning_20261016/label_sentences.py.
Code fences and headings remain in replay history but are not rating targets.
"""
import re
import pysbd


def segment(text):
    """Preserve code blocks; split paragraphs/list items into exact source spans."""
    splitter = pysbd.Segmenter(language='en', clean=False, char_span=True)
    blocks = []
    start = 0
    # An unfinished fence consumes the remainder, including truncated code.
    for match in re.finditer(r'(?m)^[ \t]*(```|~~~)[^\n]*(?:\n|$)', text):
        if match.start() < start:
            continue
        if match.start() > start:
            blocks.append(('prose', start, match.start()))
        fence = match.group(1)
        close = re.search(r'(?m)^[ \t]*' + re.escape(fence) + r'[ \t]*(?:\n|$)', text[match.end():])
        end = match.end() + close.end() if close else len(text)
        blocks.append(('code_block', match.start(), end))
        start = end
    if start < len(text):
        blocks.append(('prose', start, len(text)))
    units = []

    def add(kind, lo, hi):
        while lo < hi and text[lo].isspace():
            lo += 1
        while hi > lo and text[hi - 1].isspace():
            hi -= 1
        if lo < hi:
            units.append(dict(unit_id=len(units), kind=kind, char_start=lo, char_end=hi, text=text[lo:hi]))

    for kind, lo, hi in blocks:
        if kind == 'code_block':
            add(kind, lo, hi)
            continue
        piece = text[lo:hi]
        # Keep wrapped prose together, while honoring Markdown block boundaries.
        cuts = {0, len(piece)}
        for m in re.finditer(r'\n[ \t]*\n|(?m:^[ \t]*(?:[-*+] |\d+[.)] |#{1,6} ))', piece):
            cuts.add(m.start())
            if piece[m.start():m.end()].startswith('\n'):
                cuts.add(m.end())
        for a, b in zip(sorted(cuts), sorted(cuts)[1:]):
            part = piece[a:b]
            if not part.strip():
                continue
            if re.fullmatch(r'[\s*_#\-=]+', part):
                add('formatting', lo+a, lo+b)
            elif re.match(r'^\s*(?:#{1,6} |\*\*[^\n]+\*\*\s*$)', part):
                add('heading', lo+a, lo+b)
            else:
                # Pysbd treats physical newlines as boundaries. Mask single-line
                # wraps without changing indices into the original response.
                normalized = part.replace('\n', ' ').replace('\r', ' ')
                for span in splitter.segment(normalized):
                    add('sentence_or_fragment', lo+a+span.start, lo+a+span.end)
    validate_spans(text, units)
    return units


def validate_spans(text, units):
    cursor = 0
    for i, u in enumerate(units):
        assert u['unit_id'] == i and cursor <= u['char_start'] < u['char_end'] <= len(text)
        assert not text[cursor:u['char_start']].strip(), 'Uncovered non-whitespace text'
        assert text[u['char_start']:u['char_end']] == u['text']
        cursor = u['char_end']
    assert not text[cursor:].strip(), 'Uncovered final text'
