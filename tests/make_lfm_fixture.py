#!/usr/bin/env python3
"""Create deterministic random hybrid weights for arithmetic tests, never quality evaluation."""
import argparse
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer-source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    import torch
    from transformers import Lfm2Config, Lfm2ForCausalLM
    torch.manual_seed(81597)
    torch.set_num_threads(4)
    config = Lfm2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=65536,
        layer_types=['conv', 'full_attention', 'conv'], block_auto_adjust_ff_dim=False,
        conv_L_cache=3, conv_bias=False, bos_token_id=1, eos_token_id=7,
        tie_word_embeddings=True, max_position_embeddings=128)
    model = Lfm2ForCausalLM(config).float().eval()
    with torch.no_grad():
        for name, weights in model.named_parameters():
            if weights.ndim == 1:
                weights.uniform_(0.6, 1.4)
            elif name.endswith('conv.conv.weight'):
                weights.normal_(0, .3)
            else:
                weights.normal_(0, .06)
    model.save_pretrained(a.out)
    for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']:
        shutil.copyfile(a.tokenizer_source / name, a.out / name)


if __name__ == '__main__':
    main()
