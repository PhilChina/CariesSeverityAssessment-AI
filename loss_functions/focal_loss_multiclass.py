import torch
import torch.nn.functional as F

from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1).cpu()

        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)
        # label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length
        label_onehot = label_onehot.cuda()

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * sub_pt ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


if __name__ == '__main__':
    logits = torch.rand(10, 3, 1)
    labels = torch.LongTensor([[0],[1],[2],[1],[2],[2],[1],[2],[1],[2]])
    f1 = FocalLoss(gamma=2, alpha=0.25)
    print(logits)
    print(f1(logits,labels))

    loss_fct = nn.CrossEntropyLoss()

    seq_loss = loss_fct(logits, labels)
    print(seq_loss)


### 使用说明：
##  分类问题中使用时，在计算loss时，对predicted target维度扩增
##  predicted = predicted.unsqueeze(dim=-1)
##  targets = targets.unsqueeze(dim=-1)
##  在Cifar10数据集上进行了测试

