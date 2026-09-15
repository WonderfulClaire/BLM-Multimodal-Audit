import json
import hashlib
from pathlib import Path
from PIL import Image
from scripts.make_vlm_fixture import generate


def test_fixture_extension_retains_frozen_defaults_and_disjoint_images(tmp_path):
    generate(tmp_path/'original')
    frozen=Path(__file__).resolve().parents[1]/'reports/vlm/20260915/fixture'
    payload=(frozen/'test.jsonl').read_bytes()
    assert hashlib.sha256(payload).hexdigest()=='db6a9c248d68afd3858775158d7ec8b23148632869748bd434c9c6beb3221c07'
    expected=[json.loads(x) for x in payload.splitlines()]
    actual=[json.loads(x) for x in (tmp_path/'original/test.jsonl').read_text().splitlines()]
    assert len(actual)==len(expected)
    # PNG compression bytes can vary with the platform's zlib/Pillow build.
    # Check semantic rows and decoded RGB, while still verifying archived bytes.
    for old, new in zip(expected, actual):
        assert {k:v for k,v in old.items() if k!='image_sha256'} == {k:v for k,v in new.items() if k!='image_sha256'}
        assert hashlib.sha256((frozen/old['image']).read_bytes()).hexdigest()==old['image_sha256']
        with Image.open(frozen/old['image']) as a, Image.open(tmp_path/'original'/new['image']) as b:
            assert a.size==b.size and a.convert('RGB').tobytes()==b.convert('RGB').tobytes()
    generate(tmp_path/'extension', groups=8, layout_offset=100)
    old=set();new=set()
    for root, target in [('original',old),('extension',new)]:
        groups=set()
        for split in ('train','dev','test'):
            rows=[json.loads(x) for x in (tmp_path/root/(split+'.jsonl')).read_text().splitlines()]
            current={r['group_id'] for r in rows}
            assert not current & groups
            groups |= current
            for row in rows:
                assert row['image_sha256'] not in target
                target.add(row['image_sha256'])
    assert not old & new
