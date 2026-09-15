"""Export recorded JSONL metrics to TensorBoard without inventing missing data.

Event wall time is import time; use the Step axis. No interpolation of evaluation
points, no fabricated throughput, GPU history, or learning-rate schedules.
"""
import argparse
import json
import math
from pathlib import Path


def scalar_metrics(row):
    result = {}
    for key in ('loss', 'grad_norm', 'kl', 'clip_fraction', 'group_reward_std',
                'response_tokens', 'optimizer_updated', 'skipped_equal_reward'):
        value = row.get(key)
        if isinstance(value, (float, int)) and math.isfinite(value):
            result['train/' + key] = float(value)
    for key in ('rewards', 'correctness_scores', 'efficiency_costs'):
        values = row.get(key)
        if values and all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
            result['train/' + key + '_mean'] = sum(values) / len(values)
    return result


def export(source, output):
    from torch.utils.tensorboard import SummaryWriter
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(source.read_text())
    output.mkdir(parents=True)
    for run in spec['runs']:
        name = run['name']
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Run name must stay inside output')
        with SummaryWriter(str(output / name)) as writer:
            writer.add_text('read_me', run.get('note', '') + '\n\n'
                '真实日志导入；横轴请选择 Step。Wall time 是导入时间，不代表训练耗时。'
                '评测只有实际执行过的点，不补中间曲线。', 0)
            config = run.get('config')
            if config:
                config = json.loads((source.parent / config).read_text())
                writer.add_text('configuration', '```json\n' + json.dumps(config, ensure_ascii=False, indent=2) + '\n```', 0)
            if run.get('training'):
                path = source.parent / run['training']
                for line in path.read_text().splitlines():
                    if not line.strip(): continue
                    row = json.loads(line)
                    if 'step' not in row: continue
                    step = int(row['step'])
                    for tag, value in scalar_metrics(row).items():
                        writer.add_scalar(tag, value, step)
                    if config and 'lr' in config:
                        writer.add_scalar('config/fixed_learning_rate', config['lr'], step)
            for point in run.get('evaluations', []):
                metric = json.loads((source.parent / point['path']).read_text())
                split, step = point['split'], point['step']
                for key in ('format_valid', 'joint_correct', 'all_correct_group_fraction'):
                    if key in metric: writer.add_scalar(f'eval_{split}/{key}', metric[key], step)
                for category, values in metric.get('by_category', {}).items():
                    writer.add_scalar(f'eval_{split}/class_{category}_joint_accuracy', values['joint_accuracy'], step)
            for item in run.get('predictions', []):
                rows = [json.loads(x) for x in (source.parent/item['path']).read_text().splitlines() if x.strip()]
                for index, row in enumerate(rows):
                    writer.add_text(f"examples_{item['split']}/sample_{index:02d}",
                        '**ID** ' + row['id'] + '\n\n**实际输出**\n\n```\n' + row['text'] +
                        '\n```\n\n**标签**\n\n```json\n' + json.dumps(row['ground_truth'], ensure_ascii=False) + '\n```', item['step'])
    (output / 'source-spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.spec, args.output)
