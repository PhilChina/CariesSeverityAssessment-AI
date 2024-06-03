import math
import os
import torch.nn.functional
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision import transforms
from skimage.transform import resize
from tqdm import tqdm

from label_info import label_dict
from utils.io import get_files_name, get_medical_image, get_json
from utils.pre_processing import fill_3d_image, norm_zero_one, norm_z_score, histogramEqualization

gpus = [0]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))


def min_contours_distance(contours_1, contours_2):
    min_distance = 10000000
    for contour_1 in contours_1:
        for contour_2 in contours_2:
            for x1, y1 in contour_1:
                for x2, y2 in contour_2:

                    dis = math.sqrt(math.pow(abs(x1 - x2), 2) + math.pow(abs(y1 - y2), 2))
                    if dis < min_distance:
                        min_distance = dis

    return min_distance

def get_bbox_from_3D_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def get_bbox_from_2D_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minxidx = int(np.min(mask_voxel_coords[0]))
    maxxidx = int(np.max(mask_voxel_coords[0])) + 1
    minyidx = int(np.min(mask_voxel_coords[1]))
    maxyidx = int(np.max(mask_voxel_coords[1])) + 1
    return [[minxidx, maxxidx], [minyidx, maxyidx]]


class Caries_Classifier_Dataset(Dataset):
    def __init__(self, image_dir, train, isTrain, K):
        super(Caries_Classifier_Dataset, self).__init__()
        self.image_dir = image_dir
        self.train = train

        k_dict = get_json('./k-floder.json')
        files = k_dict['{}'.format(K)][isTrain]

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], p=1.0)

        self.images = []  # 所有图像
        self.caries = []  # 所有mask
        self.pulp = []  #
        self.labels = []  # 所有标签

        for file in files:
            image_view, origin, spacing, direction, image_type = get_medical_image(os.path.join(self.image_dir, file, 'image.nrrd'))
            caries_view, origin, spacing, direction, image_type = get_medical_image(os.path.join(self.image_dir, file, 'label.nrrd'))

            # caries_view = np.where(caries_view+image_view > 1, 1, 0)

            # ## 参考nnUnet进行截断，保留（0.5，0.95）范围内的
            # lower_bound = np.percentile(image_view, 5)
            # upper_bound = np.percentile(image_view, 95)
            # image_view = np.clip(image_view, a_min=lower_bound, a_max=upper_bound)

            # ## 均衡化，对于背景，会生成-1049这种比较小的值
            # image_view = histogramEqualization(image_view, alpha=1, beta=2, radius=1)


            if file.find('1') == -1 and file.find('2') == -1:  # 目标文件名没有1 or 2, 这是下牙，牙冠在上部【1/2，1】
                image_view = image_view[int(image_view.shape[0] * 1 / 2):, :, :]
                caries_view = caries_view[int(caries_view.shape[0] * 1 / 2):, :, :]
            else:
                image_view = image_view[:int(image_view.shape[0] / 2), :, :]
                caries_view = caries_view[:int(caries_view.shape[0] / 2), :, :]

            image_view = np.transpose(image_view, axes=(2, 0, 1))
            caries_view = np.transpose(caries_view, axes=(2, 0, 1))

            for i in range(len(image_view)):
                if np.sum(caries_view[i]) > 15:
                    reshape_image = resize(image_view[i], output_shape=(240, 300), order=3, preserve_range=True)
                    reshape_caries = resize(caries_view[i], output_shape=(240, 300), order=0, preserve_range=True)
                    self.images.append(np.asarray(reshape_image).astype(np.float32))
                    self.caries.append(np.asarray(reshape_caries).astype(np.uint8))
                    self.labels.append(label_dict[file])

    def __getitem__(self, index):
        image = self.images[index]   ## image.dtype=np.float64
        caries = self.caries[index]  ## caries.dtype = np.uint8
        label = self.labels[index]   ## int

        augmented_image = np.asarray(image[np.newaxis, :, :])
        augmented_caries = np.asarray(caries[np.newaxis, :, :])

        if self.train:
            ## be careful! one mask->mask;  multi-mask-> masks, masks=[mask1, mask2]
            masks = [caries, caries]
            tranformed = self.transform(image=(np.asarray(image[:, :, np.newaxis])), masks=masks)
            augmented_image, augmented_masks = tranformed['image'], tranformed['masks']
            augmented_caries = augmented_masks[0]
            augmented_pulp = augmented_masks[1]

            augmented_image = np.asarray(augmented_image).squeeze()
            augmented_image = augmented_image[np.newaxis, :, :]
            augmented_caries = np.asarray(augmented_caries[np.newaxis, :, :])
            augmented_pulp = np.asarray(augmented_pulp[np.newaxis, :, :])

        if augmented_image.max(initial=None) - augmented_image.min(initial=None):
            augmented_image = norm_zero_one(augmented_image)

        return np.asarray(augmented_image), label

    def __len__(self):
        return len(self.images)


class KD_Caries_Classifier_Dataset(Dataset):
    def __init__(self, image_dir, train, isTrain, K, scale_rate):
        super(KD_Caries_Classifier_Dataset, self).__init__()
        self.image_dir = image_dir
        self.train = train
        self.scale_rate = scale_rate
        # files = get_files_name(self.image_dir)  # 获取到所有病例信息
        k_dict = get_json('./k-floder.json')
        files = k_dict['{}'.format(K)][isTrain]

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], p=1.0)

        self.images = []  # 所有图像
        self.caries = []  # 所有mask
        self.pulp = []  #
        self.labels = []  # 所有标签

        for file in files:
            image_view, origin, spacing, direction, image_type = get_medical_image(os.path.join(self.image_dir, file, 'image_mask_recolor.nrrd'))
            caries_view, origin, spacing, direction, image_type = get_medical_image(os.path.join(self.image_dir, file, 'label.nrrd'))
            pulp_view, origin, spacing, direction, image_type = get_medical_image(os.path.join(self.image_dir, file, 'pulp_seg_result_4.nrrd'))

            if file.find('1') == -1 and file.find('2') == -1:  # 目标文件名没有1 or 2, 这是下牙，牙冠在上部【1/2，1】
                image_view = image_view[int(image_view.shape[0] * 1 / 2):, :, :]
                caries_view = caries_view[int(caries_view.shape[0] * 1 / 2):, :, :]
                pulp_view = pulp_view[int(pulp_view.shape[0] * 1 / 2):, :, :]
            else:
                image_view = image_view[:int(image_view.shape[0] / 2), :, :]
                caries_view = caries_view[:int(caries_view.shape[0] / 2), :, :]
                pulp_view = pulp_view[:int(pulp_view.shape[0] / 2), :, :]

            image_view = np.transpose(image_view, axes=(2, 0, 1))
            caries_view = np.transpose(caries_view, axes=(2, 0, 1))
            pulp_view = np.transpose(pulp_view, axes=(2, 0, 1))

            for i in range(len(image_view)):
                if np.sum(caries_view[i]) > 5 and np.sum(pulp_view[i]) > 50:
                    reshape_image = resize(image_view[i], output_shape=(200, 200))  ## 这里直接换成了200 * 200，后面不作增广；
                    reshape_caries = resize(caries_view[i], output_shape=(200, 200))
                    reshape_pulp = resize(pulp_view[i], output_shape=(200, 200))
                    self.images.append(np.asarray(reshape_image).astype(np.float32))
                    self.caries.append(np.asarray(reshape_caries > reshape_caries.min()).astype(np.uint8))
                    self.pulp.append(np.asarray(reshape_pulp > reshape_pulp.min()).astype(np.uint8))
                    self.labels.append(label_dict[file])

        self.images = fill_3d_image(np.asarray(self.images), [200, 200], content=0)
        self.caries = fill_3d_image(np.asarray(self.caries), [200, 200], content=0)
        self.pulp = fill_3d_image(np.asarray(self.pulp), [200, 200], content=0)

    def __getitem__(self, index):
        # image = self.images[index//self.scale_rate]
        # caries = self.caries[index//self.scale_rate]

        image = self.images[index]  ## image.dtype=np.float64
        caries = self.caries[index]  ## caries.dtype = np.uint8
        pulp = self.pulp[index]
        label = self.labels[index]  ## int

        caries_contour = skimage.measure.find_contours(caries, 0.5)
        pulp_contour = skimage.measure.find_contours(pulp, 0.5)

        min_distance = 9999999

        if caries_contour is not None and pulp_contour is not None:
            if caries_contour[0] is not None and pulp_contour[0] is not None:
                min_distance = min_contours_distance(caries_contour, pulp_contour)
        else:
            min_distance = 90

        augmented_image = np.asarray(image[np.newaxis, :, :])
        augmented_caries = np.asarray(caries[np.newaxis, :, :])
        augmented_pulp = np.asarray(pulp[np.newaxis, :, :])

        if self.train:
            ## be careful! one mask->mask;  multi-mask-> masks, masks=[mask1, mask2]
            masks = [caries, pulp]
            tranformed = self.transform(image=(np.asarray(image[:, :, np.newaxis])), masks=masks)
            augmented_image, augmented_masks = tranformed['image'], tranformed['masks']
            augmented_caries = augmented_masks[0]
            augmented_pulp = augmented_masks[1]

            caries_contour = skimage.measure.find_contours(augmented_caries, 0.5)
            pulp_contour = skimage.measure.find_contours(augmented_pulp, 0.5)
            if caries_contour is not None and pulp_contour is not None:
                min_distance = min_contours_distance(caries_contour, pulp_contour)
            else:
                min_distance = 90

            augmented_image = np.asarray(augmented_image).squeeze()
            augmented_image = augmented_image[np.newaxis, :, :]
            augmented_caries = np.asarray(augmented_caries[np.newaxis, :, :])
            augmented_pulp = np.asarray(augmented_pulp[np.newaxis, :, :])

        if augmented_image.max(initial=None) - augmented_image.min(initial=None):
            augmented_image = norm_zero_one(augmented_image)

        return np.asarray(augmented_image + augmented_caries * 0.5 + augmented_pulp * 0.5), np.asarray(
            augmented_caries), label, np.asarray([min_distance]).astype(np.float)

    def __len__(self):
        # return len(self.images) * self.scale_rate    ## 这里进行30倍数据增广
        return len(self.images)


class Caries_3D_Dataset(Dataset):
    """
    基于单颗牙齿进行预测
    """
    def __init__(self, image_dir, train, isTrain, K):
        super(Caries_3D_Dataset, self).__init__()
        self.image_dir = image_dir
        self.train = train

        k_dict = get_json('./split/caries_classifer_split_12_30.json')
        files = k_dict['{}'.format(K)][isTrain]

        self.files = files

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], p=1.0)

        self.images = list()
        self.labels = list()

        for file in files:
            image, origin, spacing, direction, image_type = get_medical_image(os.path.join(self.image_dir, file, 'image.nrrd'))
            target = label_dict[file]
            self.images.append(image)
            self.labels.append(target)

    def __getitem__(self, index):

        image = self.images[index]
        target = self.labels[index]

        ## transform
        image = resize(image, output_shape=[160, 128, 128], order=3, preserve_range=True)

        # ## 参考nnUnet进行截断，保留（0.5，0.95）范围内的
        lower_bound = np.percentile(image, 5)
        upper_bound = np.percentile(image, 95)
        image = np.clip(image, a_min=lower_bound, a_max=upper_bound)
        image = norm_zero_one(image)
        img = np.stack([image], axis=0)  ## size: 1*d*h*w

        return img, target

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    # paramers
    LEARNING_RATE = 0.0001
    batch_size = 10
    epochs = 200
    K = 0

    # val data
    # val_data_dir = './dataset/12_10_train_data/'
    # val_dataset = Caries_Classifier_Dataset(val_data_dir, train=False, isTrain='test', K=K)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, pin_memory=False, num_workers=1)

    # 测试image
    # for i_iter, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
    #     image, label = data
    #     image = image.numpy().squeeze().squeeze()
    #     print(image.shape)
    #
    #     plt.imshow(image, cmap='gray')
    #     plt.show()


    ##  测试3d image
    test_dir = './dataset/12_27_train_data'
    test_dataset = Caries_3D_Dataset(test_dir, train=False, isTrain='test', K=K)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

    for data in test_dataloader:
        print(data[0].shape)

    ### ****************** 测试标签  ************
    # # nn_model = res2net50_v1b_26w_4s(num_classes=3)
    # nn_model = resnet18()
    # nn_model = weight_init_kaiming(nn_model)
    # # checkpoints = './checkpoints/K0-view2: ACC-0.685-Epoch-83-cp-2021-08-31-09-21-01.pth'
    # # view2_model.load_state_dict(torch.load(checkpoints))
    #
    # ## 使用多GPU加速训练
    # # 使用 nn.DataParallel
    # nn_model = nn.DataParallel(nn_model.cuda(), device_ids=gpus, output_device=gpus[0])
    #
    # nn_model.train()
    # for i_iter, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
    #     image, label = data
    #     image = image.float().cuda(non_blocking=True)
    #
    #     ## [1, 2, 0] 保存一个Batchsize的数组，里面的数字代表具体类别序号；
    #     #  torch.LongTensor([1, 0, 2])
    #     label = label.long().cuda(non_blocking=True)
    #     predict = nn_model(image)
    #     predict = torch.nn.functional.softmax(predict, dim=-1)
    #     print(predict)



