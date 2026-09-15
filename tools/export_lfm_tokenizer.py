#!/usr/bin/env python3
"""Export the pinned LFM2.5 GPT4-style byte-BPE tokenizer as SLMTOK version 2.

Token flags: 1 = skip on decode; 2 = literal added-token match. Padded model
vocabulary entries are empty, skip-marked, and never matched during encoding.
No BOS is inserted by this format or by slm_encode; chat code owns framing.
"""
import argparse
import json
from pathlib import Path
import struct

GPT4_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def unicode_ranges():
    # Use the very same HF tokenizers regex implementation as the source
    # pretokenizer, avoiding Python unicodedata/Rust Unicode-version drift.
    from tokenizers import Regex, pre_tokenizers
    text = ''.join(chr(cp) for cp in range(0x110000) if not 0xd800 <= cp <= 0xdfff)
    flags = bytearray(0x110000)
    for expression, flag in ((r'\p{L}+', 1), (r'\p{N}+', 2), (r'\s+', 4)):
        splitter = pre_tokenizers.Split(Regex(expression), behavior='removed', invert=True)
        for value, (start, end) in splitter.pre_tokenize_str(text):
            assert len(value) == end - start
            start += 0x800 if start >= 0xd800 else 0
            end += 0x800 if end > 0xd800 else 0
            for cp in range(start, end):
                flags[cp] |= flag
    ranges, start, previous = [], 0, 0
    for cp in range(0x110001):
        flag = flags[cp] if cp < len(flags) else 0
        if flag != previous:
            if previous:
                ranges.append((start, cp - 1, previous))
            start, previous = cp, flag
    return ranges


def export_tokenizer(folder, output, model_vocab=None):
    import tokenizers
    folder, output = Path(folder), Path(output)
    data = json.loads((folder / 'tokenizer.json').read_text())
    expected = {'type': 'Sequence', 'pretokenizers': [
        {'type': 'Split', 'pattern': {'Regex': GPT4_PATTERN}, 'behavior': 'Isolated', 'invert': False},
        {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': True, 'use_regex': False}]}
    if data.get('normalizer') is not None or data['pre_tokenizer'] != expected:
        raise ValueError('Expected unnormalized LFM2.5 GPT4-pattern Split + ByteLevel tokenizer')
    model = data['model']
    if model['type'] != 'BPE' or model.get('dropout') or model.get('unk_token') is not None or \
            model.get('continuing_subword_prefix') or model.get('end_of_word_suffix') or \
            model.get('byte_fallback') or model.get('ignore_merges'):
        raise ValueError('Unsupported BPE options')
    vocab = model['vocab']
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError('Base vocabulary IDs must be contiguous')
    added = {}
    for item in data['added_tokens']:
        if item['id'] in added or not item['content'] or any(item.get(k) for k in ('single_word', 'lstrip', 'rstrip')):
            raise ValueError('Unsupported or duplicate added token')
        # normalized=True is equivalent here because normalizer is absent.
        added[item['id']] = item
    required = max(max(vocab.values()), max(added, default=-1)) + 1
    size = required if model_vocab is None else model_vocab
    if not isinstance(size, int) or not required <= size <= 1000000:
        raise ValueError('Model vocabulary cannot omit tokenizer IDs')
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    decoder = {chr(byte): byte for byte in visible}
    extra = 0
    for byte in range(256):
        if byte not in visible:
            decoder[chr(256 + extra)] = byte
            extra += 1
    records = [(b'', 1) for _ in range(size)]
    for text, token in vocab.items():
        if token not in added:
            records[token] = (bytes(decoder[char] for char in text), 0)
    for token, item in added.items():
        records[token] = (item['content'].encode('utf-8'), 2 | int(item.get('special', False)))
    merges = []
    for pair in model['merges']:
        left, right = pair.split(' ') if isinstance(pair, str) else pair
        merges.append((vocab[left], vocab[right], vocab[left + right]))
    ranges = unicode_ranges()
    with output.open('wb') as stream:
        stream.write(struct.pack('<8s4I', b'SLMTOK1\0', 2, size, len(merges), len(ranges)))
        for raw, flags in records:
            stream.write(struct.pack('<2I', len(raw), flags))
            stream.write(raw)
        for row in merges + ranges:
            stream.write(struct.pack('<3I', *row))
    return {'format_version': 2, 'vocab': size, 'source_vocab': required,
            'padded_tokens': size - required, 'merges': len(merges),
            'added_tokens': len(added), 'special_tokens': sum(bool(v.get('special')) for v in added.values()),
            'unicode_ranges': len(ranges), 'tokenizers_version': tokenizers.__version__,
            'tokenizer_bytes': output.stat().st_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model-vocab', type=int)
    args = parser.parse_args()
    print(json.dumps(export_tokenizer(args.folder, args.output, args.model_vocab), indent=2))
