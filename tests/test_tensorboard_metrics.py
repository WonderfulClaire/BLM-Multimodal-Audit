from scripts.export_tensorboard import scalar_metrics


def test_metric_export_keeps_skips_and_never_invents_unlogged_values():
    values = scalar_metrics({'step': 3, 'rewards': [0., 1.], 'optimizer_updated': False,
                             'grad_norm': float('nan')})
    assert values == {'train/rewards_mean': .5, 'train/optimizer_updated': 0.}
    assert 'train/loss' not in values and 'train/kl' not in values
