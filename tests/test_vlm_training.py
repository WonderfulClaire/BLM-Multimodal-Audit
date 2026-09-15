import json
import torch
from torch import nn
from types import SimpleNamespace


def test_strict_audit_rejects_shortcuts_and_wrong_types():
    from grpo_post_training.vlm import score_answer, strict_audit
    truth={'risk_category':'risk','risk_level':'high','reason':'red square'}
    text=json.dumps(truth)
    assert score_answer(text, truth)['reward'] == 1
    assert score_answer('prefix '+text, truth)['reward'] == 0
    assert strict_audit('{"risk_category":[],"risk_level":"high","reason":"x"}') is None
    assert strict_audit('{"risk_category":"risk","risk_category":"normal","risk_level":"high","reason":"x"}') is None
    assert score_answer(json.dumps({**truth,'risk_category':'normal'}),truth)['reward'] == 0


def test_response_mask_preserves_image_inputs_and_excludes_prompt():
    from grpo_post_training.vlm import response_log_probs
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.logits=nn.Parameter(torch.zeros(1,4,10))
        def forward(self, **kwargs):
            self.kwargs=kwargs
            return SimpleNamespace(logits=self.logits)
    m=Model()
    inputs={'input_ids':torch.tensor([[7,8,9]]),'attention_mask':torch.ones(1,3,dtype=torch.long),
            'pixel_values':torch.randn(4,12),'image_grid_thw':torch.tensor([[1,2,2]])}
    actions=torch.tensor([[1,2]])
    probs=response_log_probs(m,inputs,actions)
    assert probs.shape == (2,)
    (-probs.mean()).backward()
    assert m.logits.grad[:,:2].abs().sum() == 0
    assert m.logits.grad[:,2:].abs().sum() > 0
    assert m.kwargs['pixel_values'] is inputs['pixel_values']
    assert torch.equal(m.kwargs['image_grid_thw'],inputs['image_grid_thw'])
    assert m.kwargs['attention_mask'].shape == m.kwargs['input_ids'].shape == (1,4)


def test_model_messages_do_not_include_labels_or_ids():
    from grpo_post_training.vlm import messages_for
    row={'id':'answer-leak-id','prompt':'Follow the policy.', 'ground_truth':{'reason':'secret-answer'}}
    text=json.dumps(messages_for(row))
    assert 'answer-leak-id' not in text and 'secret-answer' not in text
    assert 'Follow the policy.' in text

def test_vlm_review_queue_never_mines_heldout_or_auto_approves():
    from grpo_post_training.vlm import review_failure
    import pytest
    row={'id':'x','split':'train','group_id':'scene','image':'x.png','image_sha256':'abc',
         'prompt':'policy','rule_version':'v1'}
    score={'reward':0.,'joint_correct':False}
    failure=review_failure(row,'incorrect answer',score,0,1)
    assert failure['status']=='needs_independent_review'
    assert failure['image_sha256']=='abc' and failure['group_id']=='scene'
    assert 'curation' not in failure and 'ground_truth' not in failure
    assert review_failure(row,'correct',{'reward':1.,'joint_correct':True},0,1) is None
    with pytest.raises(ValueError,match='training'):
        review_failure({**row,'split':'test'},'bad',score,0,1)
