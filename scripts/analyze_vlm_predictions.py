"""Post-hoc content diagnostics; never alters the strict training reward.

Allows a single JSON Markdown fence for content inspection, but not arbitrary
prose extraction. Independent manifest labels and recorded rewards are checked.
"""
import argparse
import json
import re
from pathlib import Path
from grpo_post_training.vlm import score_answer


def decode_content(text):
    match=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text.strip(),flags=re.S)
    if match: text=match[1]
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d: raise ValueError('Duplicate field')
            d[k]=v
        return d
    try: value=json.loads(text,object_pairs_hook=unique)
    except (ValueError,TypeError):return None
    return value if isinstance(value,dict) else None


def analyze(cases,predictions):
    truth={r['id']:r for r in cases};pred={r['id']:r for r in predictions}
    if not truth or len(truth)!=len(cases) or len(pred)!=len(predictions) or truth.keys()!=pred.keys():
        raise ValueError('Need complete unique paired predictions')
    strict=[];category=[];joint=[];groups={};by_category={}
    for key,row in truth.items():
        actual=pred[key];expected=row['ground_truth']
        if actual['ground_truth']!=expected:raise ValueError('Changed independent labels')
        score=score_answer(actual['text'],expected)
        if actual['reward']!=score['reward']:raise ValueError('Recorded reward does not match strict protocol')
        content=decode_content(actual['text']) or {}
        c=content.get('risk_category')==expected['risk_category']
        j=c and content.get('risk_level')==expected['risk_level']
        strict.append(score['joint_correct']);category.append(c);joint.append(j)
        groups.setdefault(row.get('group_id',key),[]).append(j)
        by_category.setdefault(expected['risk_category'],[]).append(j)
    return {'samples':len(cases),'source_groups':len(groups),
            'strict_joint_accuracy':sum(strict)/len(strict),
            'content_category_accuracy':sum(category)/len(category),
            'content_joint_accuracy':sum(joint)/len(joint),
            'content_all_correct_group_fraction':sum(all(x) for x in groups.values())/len(groups),
            'content_joint_by_category':{k:sum(v)/len(v) for k,v in by_category.items()},
            'scope':'Post-hoc diagnostic permits one JSON fence; training reward and strict acceptance unchanged; reason semantics not evaluated'}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('manifest',type=Path);p.add_argument('predictions',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    read=lambda path:[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    result=analyze(read(a.manifest),read(a.predictions));a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as out:out.write(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
