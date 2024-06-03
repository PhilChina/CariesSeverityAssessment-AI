import os

import numpy as np
import torch
from skimage.transform import resize
from sklearn.metrics import classification_report
from torch import nn
import torch.nn.functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Caries_3D_Dataset
from label_info import label_dict

from model.caries_cls_module import DeepCariesClassifer
from model.resnet_3d import weight_init_kaiming
from utils.io import get_json, get_medical_image
from utils.pre_processing import norm_zero_one

gpus = [0]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

def infer(model, val_dataloader):
    print('--------------   evalutation    ------------')
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
            pred = model(image).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(label[i].item())
                y_pred.append(pred[i].item())
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))



if __name__ == "__main__":
    k_dict = get_json('./split/caries_classifer_split_12_30.json')
    image_dir = './dataset/12_30_train_data'
 
    patch_size = 5
    patch_stride = 1
    output_type = 'multiclass'

    model = DeepCariesClassifer(
        output_type=output_type,
        num_inp_channels=1,
        num_fmap_channels=128,
        att_dim=128,
        num_classes=2,
        patch_size=patch_size,
        patch_stride=patch_stride,
        k_min=100
    )

    model = weight_init_kaiming(model)
    checkpoints = './checkpoints/Caries_classification_checkpoints/01_16_K1:Epoch-212-BestAcc-0.90625.pth'
    model.load_state_dict(torch.load(checkpoints))
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    K = 1
    isTrain = 'test'

    files = k_dict['{}'.format(K)][isTrain]
    model.eval()
    for file in files:
        image, origin, spacing, direction, image_type = get_medical_image(os.path.join(image_dir, file, 'image.nrrd'))
        target = label_dict[file]
        image = resize(image, output_shape=[160, 128, 128], order=3, preserve_range=True)

        # ## 参考nnUnet进行截断，保留（0.5，0.95）范围内的
        lower_bound = np.percentile(image, 5)
        upper_bound = np.percentile(image, 95)
        image = np.clip(image, a_min=lower_bound, a_max=upper_bound)
        image = norm_zero_one(image)

        image = np.asarray(image[np.newaxis, np.newaxis, :, :, :]).astype(np.float32)
        image = torch.Tensor(image).cuda()

        result = model(image).argmax(dim=1)
        if result.detach().cpu().numpy().squeeze() != target:
            print(file)
            print(result)
            print(target)

    # val data
    val_data_dir = './dataset/12_30_train_data/'
    val_dataset = Caries_3D_Dataset(val_data_dir, train=False, isTrain='test', K=K)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=False)

    infer(model, val_dataloader)






