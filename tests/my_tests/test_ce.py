import numpy as np
import torch
from torch.nn.functional import cross_entropy


# def my_cross_entropy(pred, label, ignore_index):
#     N, C = pred.shape
#     pred_n = pred.cpu().data.numpy()
#     label_n = label.cpu().data.numpy()
#     label_p = np.zeros((N, C), dtype=pred_n.dtype)
#     idx = np.where(label_n != ignore_index)[0]
#     label_p[idx, label_n[idx]] = 1
#     idx = np.where(label_n == ignore_index)[0]
#     label_p[idx, :] = pred_n[idx, :]
#     label_p = torch.from_numpy(label_p).to(device=pred.device)
#     return cross_entropy(pred, label_p)

def my_cross_entropy(pred, label, ignore_index):
    pred2 = pred[label != ignore_index]
    label2 = label[label != ignore_index]
    label2_p = torch.zeros_like(pred2)
    label2_p
    return cross_entropy(pred2, label2)


def test_ignore_index():
    N = 1000
    C = 5
    ignore_index = 4
    pred = torch.rand(N, C)
    label = torch.randint(C, (N,))
    # label = torch.full((N, ), ignore_index)
    ignore_num = torch.sum(label == ignore_index)
    print(f"ignore_num: {ignore_num}")
    ce1 = cross_entropy(pred, label, ignore_index=ignore_index)
    ce2 = my_cross_entropy(pred, label, ignore_index=ignore_index)
    print(ce1.item(), ce2.item())
    assert abs(ce1.item() - ce2.item()) / abs(ce1.item()) <= 1e-7


if __name__ == '__main__':
    test_ignore_index()
