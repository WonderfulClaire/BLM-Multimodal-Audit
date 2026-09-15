"""Counterfactual shape fixtures for visual-conditioning checks, not risk data.

Each group shares layout and policy; only the square color changes. Red circles
are distractors and are explicitly exempt in the policy. All splits are frozen
before model evaluation, with distinct layouts and pixel hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

POLICY = ('测试规则：只有红色方块标为风险，红色圆形不算风险。'
          '存在红色方块时risk_category为风险、risk_level为高危；'
          '否则risk_category为正常、risk_level为低危。reason说明图片中的方块颜色和判定依据。')


def generate(root):
    root = Path(root);root.mkdir(parents=True, exist_ok=False)
    hashes = set();manifest = {}
    for split, start in [('train',0),('dev',10),('test',20)]:
        rows=[]
        for layout in range(start,start+4):
            for color,cn in [('red','红色'),('blue','蓝色'),('green','绿色')]:
                image=Image.new('RGB',(224,168),(245+layout%10,)*3)
                draw=ImageDraw.Draw(image)
                dx,dy=(layout*7)%15,(layout*11)%17
                if layout%2:
                    square=[130+dx,35+dy,178+dx,83+dy];circle=[20,90-dy,64,134-dy]
                else:
                    square=[20+dx,25+dy,68+dx,73+dy];circle=[140,90-dy,184,134-dy]
                draw.rectangle(square,fill=color);draw.ellipse(circle,fill='red')
                digest=hashlib.sha256(image.tobytes()).hexdigest()
                if digest in hashes:raise AssertionError('Duplicate image across splits')
                hashes.add(digest)
                name=digest[:20]+'.png';image.save(root/name)
                truth={'risk_category':'风险' if color=='red' else '正常',
                       'risk_level':'高危' if color=='red' else '低危',
                       'reason':cn+'方块；红色圆形不计入风险。'}
                rows.append({'id':digest[:20], 'image':name, 'split':split,
                             'group_id':f'layout-{layout}', 'prompt':POLICY,
                             'ground_truth':truth, 'source':'counterfactual_shape_renderer',
                             'rule_version':'red-square-exemption-v1',
                             'image_sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()})
        payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        (root/(split+'.jsonl')).write_text(payload)
        manifest[split]={'samples':len(rows),'groups':4,'sha256':hashlib.sha256(payload.encode()).hexdigest()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    print(json.dumps(generate(p.parse_args().output),indent=2))
