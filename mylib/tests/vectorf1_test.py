from mylib.vectorf1 import *


def test_f1():
    f1metric = VectorF1(unmatched_index=0)

    # this should get 0 f1.
    gold = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    pred = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])

    f1metric.reset()
    f1metric(pred, gold)
    p, r, f1 = f1metric.get_metric()
    assert abs(f1 - 0.0) < 1e-5

    # this should get 1 f1.
    gold = torch.tensor([[1, 1, 0, 1]])
    pred = torch.tensor([[1, 1, 0, 1]])

    f1metric.reset()
    f1metric(pred, gold)
    p, r, f1 = f1metric.get_metric()
    assert abs(f1 - 1.0) < 1e-5


    # this should get 0.666 f1.
    gold = torch.tensor([[1, 1, 0, 1]])
    pred = torch.tensor([[1, 0, 0, 1]])

    f1metric.reset()
    f1metric(pred, gold)
    p, r, f1 = f1metric.get_metric()
    assert abs(f1 - 0.6666666) < 1e-5




