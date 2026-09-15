import json
import pytest


def test_content_diagnostic_does_not_change_strict_reward():
    from scripts.analyze_vlm_predictions import analyze
    truth={'risk_category':'risk','risk_level':'high','reason':'red square'}
    cases=[{'id':'a','ground_truth':truth,'group_id':'scene'}]
    predictions=[{'id':'a','ground_truth':truth,'text':'```json\n'+json.dumps(truth)+'\n```','reward':0.}]
    result=analyze(cases,predictions)
    assert result['strict_joint_accuracy']==0
    assert result['content_category_accuracy']==1
    assert result['content_joint_accuracy']==1
    predictions[0]['reward']=1
    with pytest.raises(ValueError,match='reward'):
        analyze(cases,predictions)


def test_content_diagnostic_rejects_duplicate_labels():
    from scripts.analyze_vlm_predictions import decode_content
    assert decode_content('{"risk_category":"risk","risk_category":"normal"}') is None
    assert decode_content('prose {"risk_category":"risk"}') is None
