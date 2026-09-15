import pytest

def normal_case():
    return {'sections': {
        'mobility': ['Test vehicle speed: 20 km/h. Maximum permitted speed: 40 km/h.'],
        'antenna': ['Serving antenna downtilt: 5 degrees. Far-end coverage is adequate.'],
        'cell_relation': ['Serving-cell distance: 0.4 km; serving RSRP: -90 dBm.',
                          'The non-colocated neighbor uses a different carrier; interference is absent.',
                          'Serving PCI: 10; neighbor PCI: 11.'],
        'handover': ['Handover events per minute: 1. No ping-pong events.',
                     'A3 threshold: 3 dB. The threshold is correctly configured.'],
        'resource': ['Average scheduled resource blocks: 200; required minimum: 160.']}}

def test_labels_are_recomputed_from_observations():
    from data_flywheel.numeric_audit import validate_measured_case
    case = normal_case()
    assert validate_measured_case(case) == []
    case['sections']['mobility'][0] = 'Test vehicle speed: 51 km/h. Maximum permitted speed: 40 km/h.'
    case['sections']['resource'][0] = 'Average scheduled resource blocks: 120; required minimum: 160.'
    case['ground_truth'] = ['C4']
    case['audit_measurements'] = {'speed': 0}
    assert validate_measured_case(case) == ['C1', 'C8']
    case['sections']['mobility'][0] = 'Speed measurement is unavailable.'
    with pytest.raises(ValueError, match='Unrecognized'):
        validate_measured_case(case)

def test_measurements_reject_ambiguous_or_contradictory_records():
    from data_flywheel.numeric_audit import validate_measured_case
    case = normal_case()
    case['sections']['resource'].append('Average scheduled resource blocks: 90; required minimum: 160.')
    with pytest.raises(ValueError): validate_measured_case(case)

def test_numeric_flywheel_mode_rejects_wrong_teacher_labels(tmp_path):
    import copy, json
    from scripts.run_agent_flywheel import prepare
    inputs = tmp_path/'inputs';inputs.mkdir()
    case = normal_case()
    case['sections']['mobility'][0] = 'Test vehicle speed: 51 km/h. Maximum permitted speed: 40 km/h.'
    base = {'id':'base', 'split':'train', 'symptom':'Diagnose.', 'case':copy.deepcopy(case), 'ground_truth':['C1']}
    case['sections']['resource'][0] = 'Average scheduled resource blocks: 120; required minimum: 160.'
    candidate = {**base, 'id':'candidate', 'case':case, 'ground_truth':['C1','C8']}
    holdout = {**base, 'id':'heldout', 'split':'test', 'case':normal_case()}
    traces = [{'step':0,'case_id':'candidate','split':'train','reward':reward,
               'trace':[{'info':{'tool_calls':[{'predicted_root_causes':pred, 'ground_truth':['C1','C8']}]}}]}
              for reward,pred in [(.3,['C1']),(.9,['C1','C8'])]]
    for name,rows in [('train',[base]),('eval12',[holdout]),('compound_train',[candidate]),('trajectories',traces)]:
        (inputs/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    prepare(inputs,tmp_path/'valid',validator_name='measured')
    assert json.loads((tmp_path/'valid/reviews.jsonl').read_text())['decision'] == 'accept'
    candidate['ground_truth']=['C4']
    (inputs/'compound_train.jsonl').write_text(json.dumps(candidate)+'\n')
    prepare(inputs,tmp_path/'invalid',validator_name='measured')
    assert json.loads((tmp_path/'invalid/reviews.jsonl').read_text())['decision'] == 'reject'
    for row in traces:
        row['trace'][0]['info']['tool_calls'][0]['predicted_root_causes']=['C1','C8']
    (inputs/'trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in traces))
    prepare(inputs,tmp_path/'reward-only',validator_name='measured')
    assert (tmp_path/'reward-only/candidates.jsonl').read_text() == ''
