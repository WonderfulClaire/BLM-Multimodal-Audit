import json
from scripts.make_vlm_fixture import generate


def test_fixture_extension_retains_frozen_defaults_and_disjoint_images(tmp_path):
    first=generate(tmp_path/'original')
    assert first['test']['sha256']=='db6a9c248d68afd3858775158d7ec8b23148632869748bd434c9c6beb3221c07'
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
