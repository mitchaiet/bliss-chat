#!/usr/bin/env python3
"""Export HF SmolLM2/LFM2 weights and exact byte-BPE data for slm_run.

SLM weights use tied embeddings once, FP32 norms, groupwise symmetric Q4/Q8;
Q4 stores consecutive values in low/high nibbles after adding 8. FP32 scales.
No permutation: RoPE uses the native HF split-half layout in the C runner.
"""
import argparse
import json
from pathlib import Path
import struct
import unicodedata


def export_tokenizer(folder, output):
    t = json.loads((folder / "tokenizer.json").read_text())
    expected = {"type": "Sequence", "pretokenizers": [
        {"type": "Digits", "individual_digits": True},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}]}
    if t["pre_tokenizer"] != expected or t.get("normalizer") is not None:
        raise ValueError("Unsupported tokenizer: expected SmolLM2 Digits + GPT2 ByteLevel")
    vocab = t["model"]["vocab"]
    specials = {x["id"]: x["content"] for x in t["added_tokens"] if x.get("special")}
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + len(cs) - 188)
    # GPT2 initially has 188 visible bytes; subsequent codepoints start at 256.
    decoder = dict(zip(map(chr, cs), bs))
    byid = [None] * len(vocab)
    for text, i in vocab.items():
        byid[i] = text.encode() if i in specials else bytes(decoder[c] for c in text)
    merges = []
    for pair in t["model"]["merges"]:
        a, b = pair.split(" ") if isinstance(pair, str) else pair
        merges.append((vocab[a], vocab[b], vocab[a + b]))
    ranges = []
    last = 0
    lo = 0
    # Rust char::is_numeric is Unicode category N; whitespace follows White_Space.
    whites = set(range(9, 14)) | {32, 0x85, 0xa0, 0x1680, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000} | set(range(0x2000, 0x200b))
    for cp in range(0x110001):
        cat = unicodedata.category(chr(cp)) if cp < 0x110000 else "Cn"
        flags = (1 if cat.startswith("L") else 0) | (2 if cat.startswith("N") else 0) | (4 if cp in whites else 0)
        if flags != last:
            if last:
                ranges.append((lo, cp - 1, last))
            lo, last = cp, flags
    with output.open("wb") as f:
        f.write(struct.pack("<8s4I", b"SLMTOK1\0", 1, len(byid), len(merges), len(ranges)))
        for i, raw in enumerate(byid):
            f.write(struct.pack("<2I", len(raw), int(i in specials)))
            f.write(raw)
        for row in merges + ranges:
            f.write(struct.pack("<3I", *row))
    return {"vocab": len(byid), "merges": len(merges), "unicode_version": unicodedata.unidata_version, "tokenizer_bytes": output.stat().st_size}


def quantize_groups(w, bits, method="maxabs"):
    """Symmetric grouped quantization, optionally minimizing weight MSE.

    MSE uses a fixed scale grid followed by alternating nearest-integer and
    least-squares updates. The original max-absolute scale is always a
    candidate, so reconstruction error cannot increase. No evaluation
    questions, activations, or calibration data enter this optimization.
    """
    import torch
    qmax = {4: 7, 6: 31, 8: 127}[bits]
    scale = w.abs().amax(dim=1) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (w / scale[:, None]).round().clamp(-qmax, qmax)
    if method == "mse":
        best_error = ((w - q * scale[:, None]) ** 2).mean(dim=1)
        best_scale = scale.clone()
        # Bound temporary allocations for the large tied embedding matrix.
        for start in range(0, w.shape[0], 16384):
            end = min(start + 16384, w.shape[0])
            part, initial = w[start:end], scale[start:end]
            for factor in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05):
                trial = initial * factor
                for _ in range(3):
                    rounded = (part / trial[:, None]).round().clamp(-qmax, qmax)
                    denom = rounded.square().sum(dim=1)
                    trial = torch.where(denom > 0,
                        (part * rounded).sum(dim=1) / denom.clamp_min(1), initial)
                rounded = (part / trial[:, None]).round().clamp(-qmax, qmax)
                error = (part - rounded * trial[:, None]).square().mean(dim=1)
                improved = error < best_error[start:end]
                best_error[start:end] = torch.minimum(best_error[start:end], error)
                best_scale[start:end] = torch.where(improved, trial, best_scale[start:end])
        scale = best_scale
        q = (w / scale[:, None]).round().clamp(-qmax, qmax)
    elif method != "maxabs":
        raise ValueError("Unknown scale method: " + method)
    return q.to(torch.int8).flatten(), scale


def export_weights(folder, output, bits, group, context, scale_method="maxabs"):
    import torch
    from safetensors import safe_open
    c = json.loads((folder / "config.json").read_text())
    arch = c["model_type"]
    if arch not in ("llama", "lfm2"):
        raise ValueError("Unsupported architecture: " + arch)
    if not c.get("tie_word_embeddings", c.get("tie_embedding", arch == "lfm2")):
        raise ValueError("Native export requires tied embeddings")
    if any(c.get(k, False) for k in ("attention_bias", "mlp_bias", "conv_bias")):
        raise ValueError("Native export requires bias-free matrices and convolutions")
    d, layers, heads, kv, vocab = (c[k] for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size"])
    hd = c.get("head_dim", d // heads)
    if bits == 6 and (arch != "lfm2" or group != 64):
        raise ValueError("Q6 requires an LFM model with group64")
    if group not in (16, 32, 64) or d % group or hd * heads != d:
        raise ValueError("The native implementation requires group16/32/64 and full-head attention")
    if not 32 <= context <= 8192 or not 1 <= layers <= 128:
        raise ValueError("context must be32..8192 and layers1..128")
    handles = [safe_open(str(p), framework="pt", device="cpu") for p in folder.glob("*.safetensors")]
    index = {k: h for h in handles for k in h.keys()}
    def shape(name):
        return tuple(index[name].get_slice(name).get_shape())
    def tensor(name):
        return index[name].get_tensor(name).float().contiguous()
    if arch == "lfm2":
        ff = shape("model.layers.0.feed_forward.w1.weight")[0]
        types = c.get("layer_types")
        if types is None or len(types) != layers or any(x not in ("conv", "full_attention") for x in types):
            raise ValueError("LFM requires explicit conv/full_attention layer_types")
        conv_width = c.get("conv_L_cache", 3)
        if conv_width != 3:
            raise ValueError("Only LFM conv width3 is supported")
        theta = c.get("rope_parameters", {}).get("rope_theta", c.get("rope_theta", 1000000.0))
        if c.get("rope_parameters", {}).get("rope_type", "default") != "default":
            raise ValueError("Unsupported LFM RoPE scaling")
        eps, start = c.get("norm_eps", 1e-5), 6
        tj = json.loads((folder / "tokenizer.json").read_text())
        start = tj["model"]["vocab"]["<|im_start|>"]
        version = 2
    else:
        ff = c["intermediate_size"]
        types, conv_width, theta, eps, start, version = ["full_attention"] * layers, 0, c["rope_theta"], c["rms_norm_eps"], c["bos_token_id"], 1
    if ff % group:
        raise ValueError("FFN width must be divisible by the group size")
    written = set()
    def write_tensor(f, name, expected, matrix):
        if shape(name) != tuple(expected):
            raise ValueError(f"Unexpected shape for {name}: {shape(name)} != {expected}")
        written.add(name)
        w = tensor(name)
        if not torch.isfinite(w).all():
            raise ValueError("Nonfinite tensor: " + name)
        if not matrix or bits == 32:
            f.write(w.numpy().astype("<f4").tobytes())
            return
        q, scale = quantize_groups(w.reshape(-1, group), bits, scale_method)
        if bits == 4:
            u = (q.to(torch.int16) + 8).to(torch.uint8)
            f.write((u[0::2] | (u[1::2] << 4)).numpy().tobytes())
        elif bits == 6:
            u = (q.to(torch.int16) + 32).to(torch.uint8).reshape(-1, 64)
            low = (u[:, 0::2] & 15) | ((u[:, 1::2] & 15) << 4)
            high = torch.zeros((u.shape[0], 16), dtype=torch.uint8)
            for lane in range(4):
                high |= (u[:, lane::4] >> 4) << (2 * lane)
            f.write(torch.cat((low, high), dim=1).flatten().numpy().tobytes())
        else:
            f.write(q.numpy().tobytes())
        f.write(scale.numpy().astype("<f4").tobytes())
    with output.open("wb") as f:
        header = bytearray(256)
        struct.pack_into("<8s14I2f", header, 0, b"SLMODEL1", version, bits, d, ff, layers, heads, kv, hd, vocab, context, group, c["bos_token_id"], c["eos_token_id"], int(arch == "lfm2"), theta, eps)
        if version == 2:
            struct.pack_into("<2I", header, 72, conv_width, start)
            header[80:80 + layers] = bytes(int(x == "conv") for x in types)
        notice_offset = max(128, 80 + layers) if version == 2 else 128
        notice = b"Modified model file: converted for Bliss Chat. See MODEL_CARD.md and MODEL-LICENSE.txt."
        if len(notice) >= 256 - notice_offset:
            notice = b"Modified for Bliss; see MODEL_CARD.md."
        header[notice_offset:notice_offset + len(notice)] = notice
        f.write(header)
        write_tensor(f, "model.embed_tokens.weight", (vocab, d), True)
        for layer in range(layers):
            p = f"model.layers.{layer}."
            norms = ["operator_norm.weight", "ffn_norm.weight"] if arch == "lfm2" else ["input_layernorm.weight", "post_attention_layernorm.weight"]
            for name in norms:
                write_tensor(f, p + name, (d,), False)
            if types[layer] == "conv":
                write_tensor(f, p + "conv.conv.weight", (d, 1, conv_width), False)
                matrices = [("conv.in_proj.weight", (3 * d, d)), ("conv.out_proj.weight", (d, d))]
            else:
                if arch == "lfm2":
                    for name in ["self_attn.q_layernorm.weight", "self_attn.k_layernorm.weight"]:
                        write_tensor(f, p + name, (hd,), False)
                out = "self_attn.out_proj.weight" if arch == "lfm2" else "self_attn.o_proj.weight"
                matrices = [("self_attn.q_proj.weight", (d, d)), ("self_attn.k_proj.weight", (kv * hd, d)), ("self_attn.v_proj.weight", (kv * hd, d)), (out, (d, d))]
            mlp = ["feed_forward.w1.weight", "feed_forward.w3.weight", "feed_forward.w2.weight"] if arch == "lfm2" else ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
            matrices += list(zip(mlp, [(ff, d), (ff, d), (d, ff)]))
            for name, dimensions in matrices:
                write_tensor(f, p + name, dimensions, True)
        write_tensor(f, "model.embedding_norm.weight" if arch == "lfm2" else "model.norm.weight", (d,), False)
    extra = set(index) - written - {"lm_head.weight"}
    if extra:
        raise ValueError("Unexported model tensors: " + repr(sorted(extra)))
    attention_layers = types.count("full_attention")
    return {"architecture": arch, "format_version": version, "bits": bits, "group": group, "scale_method": scale_method, "context": context, "model_bytes": output.stat().st_size, "kv_bytes_fp32": 2 * attention_layers * context * kv * hd * 4, "conv_state_bytes_fp32": types.count("conv") * d * conv_width * 4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="local HF snapshot or saved model directory")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--bits", type=int, choices=[4, 6, 8, 32], default=4)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--scale-method", choices=["maxabs", "mse"], default="maxabs")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--tokenizer-only", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((a.model / "config.json").read_text())
    if config["model_type"] == "lfm2":
        from export_lfm_tokenizer import export_tokenizer as export_lfm_tokenizer
        meta = export_lfm_tokenizer(a.model, a.out / "TOKENIZER.SLT", model_vocab=config["vocab_size"])
    else:
        meta = export_tokenizer(a.model, a.out / "TOKENIZER.SLT")
    if not a.tokenizer_only:
        import torch
        torch.set_num_threads(4)
        meta.update(export_weights(a.model, a.out / "MODEL.SLM", a.bits, a.group, a.context, a.scale_method))
    (a.out / "export.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
