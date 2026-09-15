#!/usr/bin/env python3
"""CPU-only LFM tokenizer/masking and tiny hybrid-model LoRA/KL audit.

Reads training messages and tokenizer/config files only; never loads model
weights or evaluation cases. Uses the supplied train_coherence.py functions.
"""
import argparse
import collections
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn.functional as F
import peft
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Lfm2Config, Lfm2ForCausalLM


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_training(path):
    spec = importlib.util.spec_from_file_location('audited_training_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def masking_audit(module, model_dir, data_path, per_source):
    # Execute only the actual tokenizer-loader block from main, ending before
    # its dataset/model/trainer setup. No checkpoint weights or CUDA work occurs.
    main_source = inspect.getsource(module.main)
    loader_code = main_source[main_source.index('    tokenizer_config = '):
                              main_source.index('    train, train_rejected = ')]
    loader_scope = {**module.__dict__, 'args': SimpleNamespace(model=str(model_dir))}
    exec(compile(textwrap.dedent(loader_code), '<audited-tokenizer-loader>', 'exec'), loader_scope)
    tokenizer = loader_scope['tokenizer']
    groups, risky = collections.defaultdict(list), []
    rows_seen = 0
    with data_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            rows_seen += 1
            source = row.get('source', '(missing)')
            fingerprint = hashlib.sha256(json.dumps(row['messages'], sort_keys=True).encode()).hexdigest()
            groups[source].append((fingerprint, row))
            if any(m.get('role') == 'assistant' and
                   (any(m.get(key) for key in ('thinking', 'reasoning', 'reasoning_content', 'tool_calls')) or
                    '</think>' in m.get('content', '') or
                    'CONTINUE_FINAL_MESSAGE_TAG ' in m.get('content', ''))
                   for m in row['messages']):
                risky.append((fingerprint, row))
    selected = {key: row for group in groups.values()
                for key, row in sorted(group, key=lambda item: item[0])[:per_source]}
    selected.update(risky)
    counts = collections.Counter()
    selected_sources = collections.Counter()
    failures = []
    for fingerprint, row in sorted(selected.items()):
        messages = row['messages']
        selected_sources[row.get('source', '(missing)')] += 1
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        item = module.encode_messages(tokenizer, messages, 512)
        if len(full) > 512:
            assert item is None, 'overlength row was not rejected'
            counts['over_512_rejected'] += 1
            continue
        prefixes_valid = True
        expected_labels = [-100] * len(full)
        for i, message in enumerate(messages):
            if message['role'] != 'assistant':
                continue
            before = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True)
            through = tokenizer.apply_chat_template(messages[:i + 1], tokenize=True, add_generation_prompt=False)
            if full[:len(before)] != before or full[:len(through)] != through:
                prefixes_valid = False
            expected_labels[len(before):len(through)] = full[len(before):len(through)]
            counts['assistant_boundaries_checked'] += 1
        if not prefixes_valid:
            counts['noninvariant_prefix_rows'] += 1
            if item is not None:
                failures.append({'messages_sha256': fingerprint, 'reason': 'noninvariant prefix accepted'})
            else:
                counts['noninvariant_prefix_safely_rejected'] += 1
            continue
        if item is None:
            failures.append({'messages_sha256': fingerprint, 'reason': 'valid supervised row unexpectedly rejected'})
            continue
        native = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                               return_dict=True, return_assistant_tokens_mask=True)
        assert native['input_ids'] == full
        native_labels = [token if flag else -100 for token, flag in zip(full, native['assistant_masks'])]
        if item['input_ids'] != full or item['labels'] != expected_labels or item['labels'] != native_labels:
            failures.append({'messages_sha256': fingerprint, 'reason': 'assistant mask differs from native generation spans'})
            continue
        assert len(item['labels']) == len(full)
        assert item['labels'][0] == -100
        assert any(label != -100 for label in item['labels'][1:])
        counts['exact_mask_rows'] += 1
        counts['supervised_tokens'] += sum(label != -100 for label in item['labels'])
    return {
        'training_rows_scanned': rows_seen, 'selected_rows': len(selected),
        'sample_per_source': per_source, 'source_rows': {k: len(v) for k, v in groups.items()},
        'selected_sources': dict(selected_sources), 'template_sensitive_rows_in_full_data': len(risky),
        'checks': dict(counts), 'failures': failures, 'tokenizer_class': type(tokenizer).__name__,
        'tokenizer_loader_checked_from_training_main': True,
        'bos_token_id': tokenizer.bos_token_id, 'eos_token_id': tokenizer.eos_token_id,
        'template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    }


def tiny_hybrid_audit(module, model_dir):
    torch.set_num_threads(2)
    torch.manual_seed(9017)
    config = Lfm2Config.from_pretrained(str(model_dir), local_files_only=True)
    # Retain native convolution settings and feed-forward adjustment behavior,
    # but make every learned tensor tiny. Never instantiate the real 350M model.
    config.vocab_size = 97
    config.hidden_size = 32
    config.block_dim = 32
    config.conv_dim = 32
    config.intermediate_size = 96
    config.block_ff_dim = 96
    config.block_multiple_of = 8
    config.num_hidden_layers = 2
    config.layer_types = ['conv', 'full_attention']
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.max_position_embeddings = 256
    config.pad_token_id, config.bos_token_id, config.eos_token_id = 0, 1, 2
    config.use_cache = False
    config._attn_implementation = 'sdpa'
    with torch.device('cpu'):
        base = Lfm2ForCausalLM(config).float()
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=['q_proj', 'v_proj', 'out_proj']))
    model.train()
    assert all(p.device.type == 'cpu' for p in model.parameters())
    adapter_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert adapter_names and all('lora_' in name for name in adapter_names)
    assert any('.conv.out_proj.' in name for name in adapter_names)
    assert any('.self_attn.q_proj.' in name for name in adapter_names)
    assert any('.self_attn.v_proj.' in name for name in adapter_names)
    features = []
    for i, length in enumerate((144, 160, 96)):
        ids = torch.randint(3, config.vocab_size, (length,)).tolist()
        labels = [token if j >= (13, 21, 9)[i] and j % 4 else -100 for j, token in enumerate(ids)]
        features.append({'input_ids': ids, 'labels': labels, 'retention': i != 1})
    batch = module.Collator(0)(features)
    assert all(t.device.type == 'cpu' for t in batch.values())
    assert torch.all(batch['labels'][batch['attention_mask'] == 0] == -100)
    model_inputs = {key: value for key, value in batch.items() if key != 'retention_mask'}
    with torch.no_grad(), model.disable_adapter():
        teacher_before = model(**{k: v for k, v in model_inputs.items() if k != 'labels'}).logits.detach().clone()
    # Default B=0 would make KL trivially zero. Perturb every adapter B to make
    # the gradient and frozen-teacher tests sensitive to an actual mismatch.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(mean=0.0, std=0.04)
    trainer = object.__new__(module.FiniteTrainer)
    trainer.kl_weight, trainer.kl_examples = 0.7, 2
    loss, result = trainer.compute_loss(model, dict(batch), return_outputs=True)
    logits = result.logits
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size),
                         batch['labels'][:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(ce, result.loss, atol=1e-6, rtol=1e-6)
    # Independently enumerate label positions, then use the preceding logits.
    coordinates = [(row, token - 1) for row in (0, 2)
                   for token in range(1, batch['labels'].shape[1])
                   if batch['labels'][row, token].item() != -100]
    assert len(coordinates) > 128, 'must exercise the position-sampling branch'
    indices = torch.linspace(0, len(coordinates) - 1, 128).long().tolist()
    coordinates = [coordinates[index] for index in indices]
    rows, positions = zip(*coordinates)
    student_logp = F.log_softmax(logits[list(rows), list(positions)].float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_before[list(rows), list(positions)].float(), dim=-1)
    manual_kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1).mean()
    expected = ce + trainer.kl_weight * manual_kl
    assert manual_kl.item() > 1e-8
    assert torch.allclose(loss, expected, atol=2e-6, rtol=1e-6), (loss.item(), expected.item())
    wrong_teacher = F.log_softmax(teacher_before[list(rows), [p + 1 for p in positions]].float(), dim=-1)
    wrong_shift_kl = (wrong_teacher.exp() * (wrong_teacher - student_logp)).sum(dim=-1).mean()
    assert abs(wrong_shift_kl.item() - manual_kl.item()) > 1e-6

    def gradient_check():
        norms = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                norm = parameter.grad.norm().item()
                assert norm > 0, name
                norms[name] = norm
            else:
                assert parameter.grad is None, name
        return norms

    loss.backward(retain_graph=True)
    combined_norms = gradient_check()
    model.zero_grad(set_to_none=True)
    ((loss - result.loss) / trainer.kl_weight).backward()
    kl_norms = gradient_check()
    with torch.no_grad(), model.disable_adapter():
        teacher_after = model(**{k: v for k, v in model_inputs.items() if k != 'labels'}).logits
    assert torch.equal(teacher_before, teacher_after), 'disabled-adapter teacher changed'
    with torch.no_grad():
        before_merge = model(**{k: v for k, v in model_inputs.items() if k != 'labels'}).logits
        merged = model.merge_and_unload()
        after_merge = merged(**{k: v for k, v in model_inputs.items() if k != 'labels'}).logits
    assert torch.allclose(before_merge, after_merge, atol=2e-6, rtol=2e-5)
    assert not torch.cuda.is_initialized()
    return {
        'device': 'cpu', 'dtype': 'float32', 'layers': config.layer_types,
        'hidden_size': config.hidden_size, 'vocab_size': config.vocab_size,
        'total_parameters': sum(p.numel() for p in merged.parameters()),
        'adapter_tensors': len(adapter_names), 'adapter_names': adapter_names,
        'assistant_kl_positions': len(coordinates), 'ce': ce.item(), 'kl': manual_kl.item(),
        'combined_loss': loss.item(), 'manual_loss_abs_error': abs(loss.item() - expected.item()),
        'wrong_shift_kl': wrong_shift_kl.item(),
        'combined_gradient_norms': combined_norms, 'kl_only_gradient_norms': kl_norms,
        'frozen_base_gradients_absent': True, 'disabled_adapter_teacher_unchanged': True,
        'merge_max_logit_abs_error': (before_merge - after_merge).abs().max().item(),
        'cuda_initialized': torch.cuda.is_initialized(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--training-script', type=Path, required=True)
    parser.add_argument('--train-jsonl', type=Path, required=True)
    parser.add_argument('--sample-per-source', type=int, default=64)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    assert args.sample_per_source > 0
    module = load_training(args.training_script)
    report = {
        'training_script_sha256': sha256(args.training_script),
        'training_jsonl_sha256': sha256(args.train_jsonl),
        'model_small_file_hashes': {name: sha256(args.model_dir / name) for name in
                                  ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')},
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__},
        'masking': masking_audit(module, args.model_dir, args.train_jsonl, args.sample_per_source),
        'tiny_hybrid': tiny_hybrid_audit(module, args.model_dir),
        'trained_weights_loaded': False, 'evaluation_cases_read': False,
    }
    if args.output:
        with args.output.open('x') as stream:
            stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert not report['masking']['failures'], 'masking audit failed; see report'


if __name__ == '__main__':
    main()
