import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from monai.transforms import AsDiscrete, Activations
from sklearn.metrics import classification_report
from monai.metrics import ROCAUCMetric
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Caries_3D_Dataset
from model.caries_cls_module import DeepCariesClassifer
from model.resnet_3d import weight_init_kaiming

gpus = [0]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

## 预定义的函数 参考：monai, 进行One-hot转换
to_onehot = AsDiscrete(to_onehot=2)
act = Activations(softmax=True)


def train(train_dataloader, val_dataloader, nn_model, criterion, optimizer, epochs, checkpoints, K):
    best_acc = 0
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    auc_metric = ROCAUCMetric()
    metric_values = list()
    epoch_num = epochs
    for epoch in range(epoch_num):
        print('-' * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        nn_model.train()
        epoch_loss = 0
        step = 0

        for i_iter, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            step += 1
            image, label = data
            image = image.float().cuda(non_blocking=True)

            ## [1, 2, 0] 保存一个Batchsize的数组，里面的数字代表具体类别序号；
            #  torch.LongTensor([1, 0, 2])

            # label = torch.squeeze(label, 1)
            label = label.long().cuda(non_blocking=True)

            optimizer.zero_grad()
            predict = nn_model(image)
            # predict = torch.nn.functional.softmax(predict, dim=1)  ## [2, -1, 0.1] -> [0.68, 0.1, 0.22] 归一化
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        scheduler.step()

        print('--------------   Val    ------------')
        
        nn_model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32)
            y_pred = y_pred.cuda()
            y = torch.tensor([], dtype=torch.long)
            y = y.cuda()

            for batch, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                image, label = data
                image = image.float().cuda(non_blocking=True)
                # label = torch.squeeze(label, 1)
                label = label.long().cuda(non_blocking=True)
                y_pred = torch.cat([y_pred, nn_model(image)], dim=0)
                y = torch.cat([y, label], dim=0)

            ## computer metric
            y_onehot = [to_onehot(i) for i in y]
            y_pred_act = [act(i) for i in y_pred]

            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(auc_result)

            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)

            if acc_metric > best_metric:
                best_metric = acc_metric
                best_metric_epoch = epoch + 1
                torch.save(nn_model.module.state_dict(),
                           os.path.join(checkpoints, "01_16_K{}:Epoch-{}-BestAcc-{}.pth".format(K, epoch, best_metric)))
                print('saved new best metric model')

            print(f"current epoch: {epoch + 1} current AUC: {auc_result:.4f}"
                  f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f}"
                  f" at epoch: {best_metric_epoch}")

        print('--------------   evalutation    ------------')
        nn_model.eval()
        y_true = list()
        y_pred = list()
        # class_names = ["浅龋", "中龋", "深龋"]
        class_names = ["中龋", "深龋"]
        with torch.no_grad():
            for batch, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                image, label = data
                image = image.float().cuda(non_blocking=True)
                # label = torch.squeeze(label, 1)
                label = label.long().cuda(non_blocking=True)
                pred = nn_model(image).argmax(dim=1)
                for i in range(len(pred)):
                    y_true.append(label[i].item())
                    y_pred.append(pred[i].item())
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    ### 保存Figure形式的数据
    plt.figure('train', (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Validation: Area under the ROC curve")
    x = [(i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    fig_path = './result_figure'
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, "K{}_visual_training_process.jpg".format(K)))
    plt.show()
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


if __name__ == "__main__":
    # paramers
    LEARNING_RATE = 0.001

    batch_size = 10
    epochs = 800
    K = 4

    alpha = 0.5
    beta = 0.5

    train_data_dir = './dataset/12_30_train_data/'
    train_dataset = Caries_3D_Dataset(train_data_dir, train=True, isTrain='train', K=K)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # val data
    val_data_dir = './dataset/12_30_train_data/'
    val_dataset = Caries_3D_Dataset(val_data_dir, train=False, isTrain='test', K=K)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=False)

    # nn_model = res2net50_v1b_26w_4s(num_classes=3)
    # nn_model = resnet152()
    # nn_model = generate_model(50, n_classes=2)  # 10, 18, 34, 50, 101, 152, 200

    patch_size = 5
    patch_stride = 1
    output_type = 'multiclass'

    nn_model = DeepCariesClassifer(
        output_type=output_type,
        num_inp_channels=1,
        num_fmap_channels=128,
        att_dim=128,
        num_classes=2,
        patch_size=patch_size,
        patch_stride=patch_stride,
        k_min=100
    )
    nn_model = weight_init_kaiming(nn_model)
    # checkpoints = './checkpoints/Caries_classification_checkpoints/K1:Epoch-10-BestAcc-0.8666666666666667.pth'
    # nn_model.load_state_dict(torch.load(checkpoints))

    ## 使用多GPU加速训练
    # 使用 nn.DataParallel
    nn_model = nn.DataParallel(nn_model.cuda(), device_ids=gpus, output_device=gpus[0])

    # cudnn.benchmark = True

    ## loss
    criterion = nn.CrossEntropyLoss()

    ## optimizer
    # momentum=0.9 代表之前10倍速度进行  动量法SGD
    # optimizer = optim.Adam(params=view2_model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(params=nn_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    checkpoints = './checkpoints/Caries_classification_checkpoints/'
    os.makedirs(checkpoints, exist_ok=True)

    train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, nn_model=nn_model, criterion=criterion,
          optimizer=optimizer, epochs=epochs, checkpoints=checkpoints, K=K)
    print('training is finished!!')
