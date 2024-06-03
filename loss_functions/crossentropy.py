import numpy as np
import torch.nn
from torch import nn, Tensor
import torch.nn.functional

from loss_functions.one_hot import convert_to_one_hot


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(prediction.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(prediction, target.long())




class ORDCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(ORDCrossEntropyLoss, self).__init__()

    def forward(self, logits, label):
        N = label.size(0)
        label = convert_to_one_hot(label, minleng=3)
        # logits = nn.functional.log_softmax(logits, dim=1)
        # loss = -torch.sum(logits * label) / N
        bceloss = nn.MSELoss(reduction='none')
        loss = bceloss(logits, label).sum()
        return loss




if __name__ == "__main__":
    critrion = ORDCrossEntropyLoss()
    predicted = torch.FloatTensor(torch.Size((10, 3))).random_() % 3
    print(predicted)
    label = torch.LongTensor(10).random_() % 3
    print(label)

    loss = critrion(predicted, label)
    print(loss)




