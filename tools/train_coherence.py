#!/usr/bin/env python3
"""Conservative, reproducible assistant-only LoRA adaptation of a local model.

Keeps the pretrained checkpoint intact; exports a separate merged candidate.
No benchmark data is consumed. Checkpoint choice uses adaptation validation.
"""
import argparse
import hashlib
import json
import os
import pathlib
import random
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('WANDB_DISABLED', 'true')

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Trainer, TrainingArguments, TrainerCallback, set_seed


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def encode_messages(tokenizer, messages, max_length):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(ids) > max_length:
        return None
    labels = [-100] * len(ids)
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prompt_prefix = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True)
        answer_prefix = tokenizer.apply_chat_template(messages[:i + 1], tokenize=True, add_generation_prompt=False)
        begin, end = len(prompt_prefix), len(answer_prefix)
        # Some templates rewrite earlier thinking/tool text when later turns
        # are present. Reject those rows instead of placing labels at offsets
        # that no longer belong to the assistant's actual response.
        if ids[:begin] != prompt_prefix or ids[:end] != answer_prefix:
            return None
        labels[begin:end] = ids[begin:end]
    if not any(label != -100 for label in labels[1:]):
        return None
    return {'input_ids': ids, 'labels': labels}


def load_examples(path, tokenizer, max_length, limit=0):
    rows, rejected = [], 0
    with open(path) as f:
        for line in f:
            source = json.loads(line)
            item = encode_messages(tokenizer, source['messages'], max_length)
            if item is None:
                rejected += 1
            else:
                # Public replay data may anchor to the unchanged teacher.
                # New corrective examples must be free to improve on it.
                retention = source.get('retention')
                if retention is not None and not isinstance(retention, bool):
                    raise ValueError('retention must be a JSON boolean')
                item['retention'] = (retention if retention is not None else
                    source.get('source') not in ('bliss-procedural-v1', 'bliss-calibration-v1'))
                rows.append(item)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError('No usable examples: ' + str(path))
    return rows, rejected


class Collator:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, features):
        width = ((max(len(x['input_ids']) for x in features) + 7) // 8) * 8
        return {
            'input_ids': torch.tensor([x['input_ids'] + [self.pad] * (width - len(x['input_ids'])) for x in features]),
            'attention_mask': torch.tensor([[1] * len(x['input_ids']) + [0] * (width - len(x['input_ids'])) for x in features]),
            'labels': torch.tensor([x['labels'] + [-100] * (width - len(x['labels'])) for x in features]),
            'retention_mask': torch.tensor([x['retention'] for x in features], dtype=torch.bool),
        }


class Progress(TrainerCallback):
    def __init__(self, path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = {'time': time.time(), 'step': state.global_step, 'epoch': state.epoch, **(logs or {})}
        with open(self.path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = True
            control.should_save = True

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if gradients and not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item():
            raise FloatingPointError('Nonfinite gradient; aborting before optimizer update.')


class FiniteTrainer(Trainer):
    kl_weight = 0.0
    kl_examples = 8

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        retention = inputs.pop('retention_mask')
        result = model(**inputs)
        loss = result.loss
        if self.kl_weight > 0 and retention.any():
            # Preserve the frozen instruction model on public retention data.
            # Only a bounded sample of assistant positions is materialized as
            # FP32 distributions, avoiding a second full-vocabulary batch.
            selected = retention.nonzero().flatten()[:self.kl_examples]
            teacher_inputs = {k: v[selected] for k, v in inputs.items() if k != 'labels'}
            with torch.no_grad(), model.disable_adapter():
                teacher_logits = model(**teacher_inputs).logits
            supervised = inputs['labels'][selected, 1:] != -100
            coords = supervised.nonzero()
            if coords.shape[0] > 128:
                index = torch.linspace(0, coords.shape[0] - 1, 128, device=coords.device).long()
                coords = coords[index]
            if coords.numel():
                student = result.logits[selected[coords[:, 0]], coords[:, 1], :].float()
                teacher = teacher_logits[coords[:, 0], coords[:, 1], :].float()
                anchor = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(student, dim=-1),
                    torch.nn.functional.softmax(teacher, dim=-1), reduction='batchmean')
                loss = loss + self.kl_weight * anchor
            del teacher_logits
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError('Nonfinite loss; aborting training/evaluation.')
        return (loss, result) if return_outputs else loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='HuggingFaceTB/SmolLM2-360M-Instruct')
    p.add_argument('--train', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=20260915)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--eval-steps', type=int, default=100)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--max-train-examples', type=int, default=0)
    p.add_argument('--max-validation-examples', type=int, default=0)
    p.add_argument('--kl-weight', type=float, default=0.0)
    p.add_argument('--kl-examples', type=int, default=8)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--targets', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    args = p.parse_args()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'training_manifest.json').exists():
        raise RuntimeError('Output already has a manifest; choose a new run to preserve prior results.')
    set_seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer_config = pathlib.Path(args.model) / 'tokenizer_config.json'
    tokenizer_kind = json.loads(tokenizer_config.read_text()).get('tokenizer_class') if tokenizer_config.exists() else None
    tokenizer_loader = PreTrainedTokenizerFast if tokenizer_kind == 'TokenizersBackend' else AutoTokenizer
    tokenizer_compat = {'extra_special_tokens': {}} if tokenizer_kind == 'TokenizersBackend' else {}
    tokenizer = tokenizer_loader.from_pretrained(args.model, local_files_only=True, trust_remote_code=False, **tokenizer_compat)
    tokenizer.pad_token = tokenizer.eos_token
    train, train_rejected = load_examples(args.train, tokenizer, args.max_length, args.max_train_examples)
    validation, val_rejected = load_examples(args.validation, tokenizer, args.max_length, args.max_validation_examples)
    random.Random(args.seed).shuffle(train)
    manifest = {
        **vars(args), 'train_sha256': sha256(args.train), 'validation_sha256': sha256(args.validation),
        'train_examples': len(train), 'validation_examples': len(validation),
        'train_rejected': train_rejected, 'validation_rejected': val_rejected,
        'train_tokens': sum(len(x['input_ids']) for x in train),
        'supervised_tokens': sum(sum(v != -100 for v in x['labels']) for x in train),
        'retention_examples': sum(x['retention'] for x in train),
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'selection': 'lowest held-out adaptation loss (assistant CE plus configured retention KL); sealed benchmark never used for gradient updates',
        'script_sha256': sha256(__file__),
    }
    (out / 'training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=args.targets.split(','))
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    options = TrainingArguments(
        output_dir=str(out / 'checkpoints'), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation, learning_rate=args.learning_rate,
        warmup_ratio=0.06, lr_scheduler_type='cosine', weight_decay=0.01,
        bf16=True, tf32=True, optim='adamw_torch_fused', max_grad_norm=0.5,
        logging_steps=10, logging_nan_inf_filter=False, eval_strategy='steps', eval_steps=args.eval_steps,
        save_strategy='steps', save_steps=args.eval_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        report_to='none', seed=args.seed, data_seed=args.seed, dataloader_num_workers=0,
        remove_unused_columns=False, label_names=['labels'], disable_tqdm=True,
        save_safetensors=True, dataloader_pin_memory=True,
    )
    trainer = FiniteTrainer(model=model, args=options, train_dataset=train, eval_dataset=validation,
        data_collator=Collator(tokenizer.pad_token_id), callbacks=[Progress(out / 'progress.jsonl')])
    trainer.kl_weight = args.kl_weight
    trainer.kl_examples = args.kl_examples
    # This override returns a mean CE(+KL), not a num_items-normalized sum.
    trainer.model_accepts_loss_kwargs = False
    initial = trainer.evaluate()
    (out / 'baseline_validation.json').write_text(json.dumps(initial, indent=2) + '\n')
    print('BASELINE_VALIDATION', json.dumps(initial), flush=True)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    peak_gpu_bytes = torch.cuda.max_memory_allocated()
    final = trainer.evaluate()
    trainer.save_model(str(out / 'adapter'))
    tokenizer.save_pretrained(out / 'adapter')
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(out / 'merged', safe_serialization=True)
    tokenizer.save_pretrained(out / 'merged')
    completion = {'training': result.metrics, 'baseline_validation': initial, 'final_validation': final,
                  'peak_training_gpu_allocated_bytes': peak_gpu_bytes,
                  'best_checkpoint': trainer.state.best_model_checkpoint, 'completed_at': time.time(),
                  'validation_improved': final['eval_loss'] < initial['eval_loss']}
    (out / 'completed.json').write_text(json.dumps(completion, indent=2) + '\n')
    print('COMPLETED', json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()
