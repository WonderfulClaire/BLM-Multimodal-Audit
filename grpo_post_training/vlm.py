"""Real Qwen2.5-VL image-conditioned LoRA SFT/GRPO and evaluation.

The binary reward checks category and level against supplied labels. Reason text
is generated and logged, but no claim of semantic reasoning verification is made.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import torch
from PIL import Image

FIELDS = {'risk_category', 'risk_level', 'reason'}
SYSTEM = ('Read the image and follow the supplied audit policy. Return only a JSON object '
          'with exactly three nonempty string fields: risk_category, risk_level, reason. '
          'Explain the observed visual evidence in reason. Do not include markdown.')


def strict_audit(text):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result: raise ValueError('Duplicate JSON field')
            result[k] = v
        return result
    try:
        result = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None
    if not isinstance(result, dict) or set(result) != FIELDS:
        return None
    return result if all(isinstance(v, str) and v.strip() for v in result.values()) else None


def score_answer(text, truth):
    prediction = strict_audit(text)
    category = bool(prediction and prediction['risk_category'] == truth['risk_category'])
    level = bool(prediction and prediction['risk_level'] == truth['risk_level'])
    return {'reward': float(category and level), 'format_valid': prediction is not None,
            'category_correct': category, 'level_correct': level,
            'joint_correct': category and level, 'prediction': prediction}


def messages_for(row):
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': row['prompt']}]}]


def response_log_probs(model, inputs, actions):
    if actions.ndim != 2 or actions.shape[0] != 1 or actions.shape[1] < 1:
        raise ValueError('Need one nonempty generated response')
    length = inputs['input_ids'].shape[1]
    ids = torch.cat((inputs['input_ids'], actions), 1)
    kwargs = {k: v for k, v in inputs.items() if k not in {'input_ids', 'attention_mask'}}
    mask = torch.cat((inputs['attention_mask'], torch.ones_like(actions)), 1)
    logits = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1],
                   use_cache=False, **kwargs).logits[:, length-1:, :].float()
    return logits.log_softmax(-1).gather(-1, actions[..., None]).flatten()


def encode(processor, row, manifest, device):
    image_path = manifest.parent / row['image']
    with Image.open(image_path) as handle:
        image = handle.convert('RGB')
    text = processor.apply_chat_template(messages_for(row), tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], padding=True, return_tensors='pt').to(device)


@torch.no_grad()
def generate(model, processor, inputs, max_tokens, sample=False):
    options = {'do_sample': sample, 'max_new_tokens': max_tokens}
    if sample: options.update(temperature=1.0, top_p=1.0, top_k=0)
    ids = model.generate(**inputs, **options)[:, inputs['input_ids'].shape[1]:]
    return ids, processor.tokenizer.decode(ids[0], skip_special_tokens=True)


def main():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['eval', 'sft', 'grpo'])
    p.add_argument('manifest', type=Path)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sft-adapter', help='Merge this frozen SFT adapter before attaching the policy adapter')
    p.add_argument('--adapter', help='Saved policy adapter, evaluation only')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--beta', type=float, default=.01)
    a = p.parse_args()
    if (a.steps < 1 or (a.mode == 'grpo' and a.group_size < 2) or a.max_tokens < 1
            or not math.isfinite(a.lr) or not math.isfinite(a.beta) or a.lr <= 0 or a.beta < 0):
        raise ValueError('Invalid training configuration')
    if a.adapter and a.mode != 'eval': raise ValueError('--adapter is evaluation only')
    if a.sft_adapter and a.mode == 'sft': raise ValueError('SFT starts from the base checkpoint; merge SFT only for GRPO or evaluation')
    if a.output.exists(): raise FileExistsError(a.output)
    rows = [json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows): raise ValueError('Empty or duplicate samples')
    if any(strict_audit(json.dumps(r['ground_truth'])) is None for r in rows):
        raise ValueError('Invalid reference audit labels')
    if a.mode != 'eval' and any(r.get('split') != 'train' for r in rows):
        raise ValueError('Training must use train-only manifests')
    if a.mode == 'eval' and any(r.get('split') not in {'train', 'dev', 'test'} for r in rows):
        raise ValueError('Evaluation requires an explicit split, including training diagnostics')
    for row in rows:
        if row.get('image_sha256') and hashlib.sha256((a.manifest.parent/row['image']).read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError('Image changed after manifest freeze')
    torch.manual_seed(a.seed);torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(a.model, use_fast=False, min_pixels=4*28*28, max_pixels=256*28*28)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                           attn_implementation='sdpa').to(a.device)
    if a.sft_adapter:
        base = PeftModel.from_pretrained(base, a.sft_adapter).merge_and_unload()
    if a.mode == 'eval':
        model = PeftModel.from_pretrained(base, a.adapter) if a.adapter else base
    else:
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
                               target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'))
    model.eval()
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = {'mode': a.mode, 'model': a.model, 'sft_adapter': a.sft_adapter, 'adapter': a.adapter,
                'seed': a.seed, 'steps': a.steps, 'lr': a.lr, 'group_size': a.group_size,
                'max_tokens': a.max_tokens, 'beta': a.beta, 'torch': str(torch.__version__),
                'manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                'image_sha256': {r['id']: hashlib.sha256((a.manifest.parent/r['image']).read_bytes()).hexdigest() for r in rows},
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'reference': 'frozen base including merged SFT; policy adapter disabled',
                'reward': 'binary exact category and level, strict JSON; reason not semantically graded',
                'scope': 'image-conditioned pretrained VLM; evaluation validity depends on supplied data'}
    (a.output/'config.json').write_text(json.dumps(metadata, indent=2))
    def record(name, row):
        with (a.output/name).open('a') as out: out.write(json.dumps(row, ensure_ascii=False)+'\n')
    if a.mode == 'eval':
        scores = []
        for row in rows:
            inputs = encode(processor, row, a.manifest, a.device)
            _, text = generate(model, processor, inputs, a.max_tokens)
            score = score_answer(text, row['ground_truth']);scores.append(score)
            record('predictions.jsonl', {'id': row['id'], 'split': row['split'], 'text': text,
                    'ground_truth': row['ground_truth'], 'context_tokens': inputs['input_ids'].shape[1],
                    'image_grid_thw': inputs['image_grid_thw'].cpu().tolist(), **score})
        metrics = {key: sum(s[key] for s in scores)/len(scores) for key in
                   ['format_valid','category_correct','level_correct','joint_correct']}
        categories = {}; groups = {}
        for row, score in zip(rows, scores):
            categories.setdefault(row['ground_truth']['risk_category'], []).append(score)
            groups.setdefault(row.get('group_id', row['id']), []).append(score['joint_correct'])
        metrics['by_category'] = {category: {'samples': len(items),
            'category_accuracy': sum(x['category_correct'] for x in items)/len(items),
            'joint_accuracy': sum(x['joint_correct'] for x in items)/len(items)}
            for category, items in categories.items()}
        metrics['all_correct_group_fraction'] = sum(all(x) for x in groups.values())/len(groups)
        metrics.update(samples=len(rows), source_groups=len(groups), splits=sorted({r['split'] for r in rows}))
        (a.output/'metrics.json').write_text(json.dumps(metrics, indent=2));print(json.dumps(metrics));return
    trainable = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().float().cpu().clone() for p in trainable]
    optim = torch.optim.AdamW(trainable, lr=a.lr)
    order = list(range(len(rows)));random.Random(a.seed).shuffle(order)
    for step in range(a.steps):
        row = rows[order[step % len(order)]]
        inputs = encode(processor, row, a.manifest, a.device)
        metric = {'step': step, 'id': row['id'], 'optimizer_updated': False}
        optim.zero_grad()
        if a.mode == 'sft':
            text = json.dumps(row['ground_truth'], ensure_ascii=False) + processor.tokenizer.eos_token
            actions = processor.tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to(a.device)
            loss = -response_log_probs(model, inputs, actions).mean()
            loss.backward();metric.update(loss=float(loss.detach()), response_tokens=actions.numel())
            update = True
        else:
            episodes=[];rewards=[]
            for member in range(a.group_size):
                actions, text = generate(model, processor, inputs, a.max_tokens, sample=True)
                score = score_answer(text, row['ground_truth']);rewards.append(score['reward'])
                episodes.append(actions)
                record('trajectories.jsonl', {'step': step, 'member': member, 'id': row['id'], 'split':'train',
                       'text':text, 'ground_truth': row['ground_truth'], 'response_tokens':actions.numel(), **score})
            reward = torch.tensor(rewards, device=a.device)
            std = reward.std(unbiased=False)
            update = float(std) > 1e-6
            metric.update(rewards=rewards, group_reward_std=float(std),
                          route='rl_ready' if update else ('mastered_replay' if min(rewards)==1 else 'teacher_or_sft_repair'))
            if update:
                advantages=(reward-reward.mean())/std
                with torch.no_grad():
                    old=[response_log_probs(model,inputs,x).detach() for x in episodes]
                    with model.disable_adapter():
                        reference=[response_log_probs(model,inputs,x).detach() for x in episodes]
                loss_value=0.;kl_value=0.
                for index,actions in enumerate(episodes):
                    new=response_log_probs(model,inputs,actions)
                    ratio=(new-old[index]).exp()
                    surrogate=torch.minimum(ratio*advantages[index],ratio.clamp(.8,1.2)*advantages[index])
                    delta=reference[index]-new
                    kl=delta.exp()-delta-1
                    loss=(-surrogate+a.beta*kl).mean()/a.group_size
                    loss.backward();loss_value+=float(loss.detach());kl_value+=float(kl.detach().mean())/a.group_size
                metric.update(loss=loss_value,kl=kl_value)
        if update:
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            optim.step();metric.update(optimizer_updated=True,grad_norm=float(norm))
        record('training.jsonl',metric);print(json.dumps(metric),flush=True)
    model.save_pretrained(a.output/'adapter');processor.save_pretrained(a.output/'processor')
    torch.save(optim.state_dict(), a.output/'optimizer.pt')
    delta=sum(float((p.detach().float().cpu()-old).square().sum()) for p,old in zip(trainable,initial))**.5
    (a.output/'summary.json').write_text(json.dumps({'parameter_delta_l2':delta,'steps':a.steps},indent=2))

if __name__ == '__main__': main()
