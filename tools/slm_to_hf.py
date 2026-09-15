#!/usr/bin/env python3
"""Restore exported SLM quantized weights to an HF CPU checkpoint for isolation tests.

Reads the export's actual scales, including alternative clipping methods.
This is an analysis checkpoint, not a new fine-tune or an XP artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def restore_slm(model, path):
    """Load actual exported matrices into an existing tied HF model on CPU."""
    import numpy as np
    import torch
    parameters = dict(model.named_parameters())
    with Path(path).open('rb') as f:
        header = f.read(256)
        magic, version, bits, d, ff, layers, heads, kv, hd, vocab, context, group, bos, eos, arch, theta, eps = struct.unpack('<8s14I2f', header[:72])
        if magic != b'SLMODEL1' or version not in (1, 2) or (version == 2 and arch != 1):
            raise ValueError('not a supported SLM model')
        hybrid = version == 2
        if hybrid != (model.config.model_type == 'lfm2'):
            raise ValueError('export/HF architecture mismatch')
        conv_width, start = struct.unpack_from('<2I', header, 72) if hybrid else (0, bos)
        layer_types = list(header[80:80+layers]) if hybrid else [0] * layers
        def read(name, shape, matrix=True):
            if tuple(parameters[name].shape) != tuple(shape):
                raise ValueError('HF tensor shape mismatch: ' + name)
            size = int(np.prod(shape))
            if not matrix or bits == 32:
                data = np.frombuffer(f.read(size * 4), dtype='<f4').copy()
            else:
                if bits == 4:
                    packed = np.frombuffer(f.read(size // 2), dtype=np.uint8)
                    data = np.empty(size, dtype=np.float32)
                    data[0::2] = (packed & 15).astype(np.int16) - 8
                    data[1::2] = (packed >> 4).astype(np.int16) - 8
                elif bits == 6:
                    if group != 64 or not hybrid:
                        raise ValueError('Q6 requires LFM/group64')
                    packed = np.frombuffer(f.read(size * 3 // 4), dtype=np.uint8).reshape(-1, 48)
                    lo, hi = packed[:, :32], packed[:, 32:]
                    low = np.stack((lo & 15, lo >> 4), axis=-1).reshape(-1, 64)
                    high = np.stack([(hi >> (2 * lane)) & 3 for lane in range(4)], axis=-1).reshape(-1, 64)
                    data = ((low | (high << 4)).astype(np.int16) - 32).astype(np.float32).flatten()
                elif bits == 8:
                    data = np.frombuffer(f.read(size), dtype=np.int8).astype(np.float32)
                else:
                    raise ValueError(bits)
                scales = np.frombuffer(f.read(size // group * 4), dtype='<f4')
                data = (data.reshape(-1, group) * scales[:, None]).flatten()
            with torch.no_grad():
                parameters[name].copy_(torch.from_numpy(data.reshape(shape)))
        read('model.embed_tokens.weight', (vocab, d))
        for layer, kind in enumerate(layer_types):
            prefix = f'model.layers.{layer}.'
            norms = ['operator_norm.weight', 'ffn_norm.weight'] if hybrid else ['input_layernorm.weight', 'post_attention_layernorm.weight']
            for name in norms:
                read(prefix + name, (d,), False)
            if kind == 1:
                read(prefix + 'conv.conv.weight', (d, 1, conv_width), False)
                matrices = [('conv.in_proj.weight', (3 * d, d)), ('conv.out_proj.weight', (d, d))]
            elif kind == 0:
                if hybrid:
                    read(prefix + 'self_attn.q_layernorm.weight', (hd,), False)
                    read(prefix + 'self_attn.k_layernorm.weight', (hd,), False)
                out = 'self_attn.out_proj.weight' if hybrid else 'self_attn.o_proj.weight'
                matrices = [('self_attn.q_proj.weight', (d, d)), ('self_attn.k_proj.weight', (kv * hd, d)), ('self_attn.v_proj.weight', (kv * hd, d)), (out, (d, d))]
            else:
                raise ValueError('unknown layer type')
            names = ['feed_forward.w1.weight', 'feed_forward.w3.weight', 'feed_forward.w2.weight'] if hybrid else ['mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']
            matrices += list(zip(names, [(ff, d), (ff, d), (d, ff)]))
            for name, shape in matrices:
                read(prefix + name, shape)
        read('model.embedding_norm.weight' if hybrid else 'model.norm.weight', (d,), False)
        if f.read(1):
            raise ValueError('trailing model bytes')
    return dict(bits=bits, group=group, vocab=vocab, d=d, version=version, architecture='lfm2' if hybrid else 'llama', context=context)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hf', type=Path, required=True)
    p.add_argument('--slm', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--embedding-bits', type=int, choices=[8, 32], help='isolated tied embedding/head precision experiment from original HF weights')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(a.hf, local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').eval()
    parameters = dict(model.named_parameters())
    original_embedding = parameters['model.embed_tokens.weight'].detach().clone() if a.embedding_bits else None
    meta = restore_slm(model, a.slm)
    bits, group, vocab, d = (meta[k] for k in ('bits', 'group', 'vocab', 'd'))
    if original_embedding is not None:
        if a.embedding_bits == 8:
            grouped = original_embedding.reshape(-1, group)
            scales = grouped.abs().amax(dim=-1, keepdim=True) / 127
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            original_embedding = ((grouped / scales).round().clamp(-127, 127) * scales).reshape(vocab, d)
        with torch.no_grad():
            parameters['model.embed_tokens.weight'].copy_(original_embedding)
    model.config.use_cache = True
    model.save_pretrained(a.out, safe_serialization=True)
    import shutil
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'chat_template.json'):
        if (a.hf / name).exists():
            shutil.copyfile(a.hf / name, a.out / name)
    h = hashlib.sha256()
    with a.slm.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    (a.out / 'slm_origin.json').write_text(json.dumps({'source_slm': str(a.slm), 'sha256': h.hexdigest(), 'bits': bits, 'embedding_bits': a.embedding_bits or bits, 'group': group, 'reconstruction_dtype': 'float32', 'activation_quantization': False}, indent=2) + '\n')
    print(a.out, flush=True)


if __name__ == '__main__':
    main()
