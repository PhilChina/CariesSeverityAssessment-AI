import os
import numpy as np
from utils.io import save_medical_image, get_medical_image, get_files_name, save_json
from utils.utils import get_k_folder_cross_validation_samples
import SimpleITK as sitk

##  ------****************  图像增强  **************-------
def histogramEqualization(image, alpha, beta, radius):
    """
    直方图均衡化-SimpleITK
    :param image: ndarray,保存图像的数组
    :param alpha: Alpha参数是用来控制结果相对于经典直方图均衡化方法结果的相似程度
    :param beta: Beta参数用来控制图像锐化程度
    :param radius: Radius用来控制直方图统计时的区域大小
    :return: ndarray
    """
    sitk_hisequal = sitk.AdaptiveHistogramEqualizationImageFilter()
    sitk_hisequal.SetAlpha(alpha)
    sitk_hisequal.SetBeta(beta)
    sitk_hisequal.SetRadius(radius)
    image = sitk.GetImageFromArray(image, isVector=False)
    image = sitk_hisequal.Execute(image)
    image = sitk.GetArrayFromImage(image)

    return image


def medianImage(image, radius):
    """
    中值滤波器-SimpleITK
    :param image: ndimage
    :param radius: Radius用来控制直方图统计时的区域大小
    :return:
    """
    sitk_mean = sitk.MedianImageFilter()
    sitk_mean.SetRadius(radius)
    image = sitk.GetImageFromArray(image, isVector=False)
    image = sitk_mean.Execute(image)
    image = sitk.GetArrayFromImage(image)

    return image

## 归一化
def norm_zero_one(array, span=None):
    """
    根据所给数组的最大值、最小值，将数组归一化到0-1
    :param span:
    :param array: 数组
    :return: array: numpy格式数组
    """
    array = np.asarray(array).astype(np.float)
    if span is None:
        mini = array.min(initial=None)
        maxi = array.max(initial=None)
    else:
        mini = span[0]
        maxi = span[1]
        array[array < mini] = mini
        array[array > maxi] = maxi

    range = maxi - mini

    def norm(x):
        return (x - mini) / range

    return np.asarray(list(map(norm, array))).astype(np.float32)


## Z-score标准化
def norm_z_score(array):
    """
    根据所给数组的均值和标准差进行归一化，归一化结果符合正态分布，即均值为0，标准差为1
    :param array: 数组
    :return: array: numpy格式数组
    """
    array = np.asarray(array).astype(np.float)
    mu = np.average(array)  ## 均值
    sigma = np.std(array)  ## 标准差

    def norm(x):
        return (x - mu) / sigma

    return np.asarray(list(map(norm, array))).astype(np.float), mu, sigma


def fill_3d_image(image, size, content=0):
    """
    填充image
    :param image: 3D图像
    :param size: 指定大小
    :param content: 填充内容
    :return: 填充之后的图像
    """
    pad_result = []
    weight, height = size[0], size[1]
    for slice in image:
        h, w = slice.shape[0], slice.shape[1]
        slice = np.pad(slice, (((height - h) // 2, (height - h) // 2 + h % 2), ((weight - w) // 2, (weight - w) // 2 + w % 2)),
                       'constant', constant_values=content)
        pad_result.append(slice)

    return np.asarray(pad_result)


def padding_solid_size(image_path, save_path, size, content=0):
    """
    将图像填充至指定大小，这里可以指定填充内容；
    :param image_path: image原始路径
    :param save_path: image保存路径
    :param size: 目标大小
    :param content: 填充的内容
    :return: None
    """

    with os.scandir(image_path) as files:
        for file in files:
            image, origin, spacing, direction, image_type = get_medical_image(os.path.join(image_path, file.name))
            image = fill_3d_image(image, size=size, content=content)
            save_medical_image(image, os.path.join(save_path, file.name), origin=origin, spacing=spacing,
                               direction=direction, type=image_type)


def trasform_1_or_2(path, save_path):
    """
    数据集有些牙齿是在上郃，有些是下郃，这里将上郃全部上下转置；根据牙位图，上郃是1 or 2, 下郃是 3 or 4；
    :param save_path: 保存路径
    :param path: 需要转置的图像路径
    :return: None
    """
    with os.scandir(path) as files:
        for file in files:
            if file.name.find('1') == -1 and file.name.find('2') == -1:  # 目标文件名没有1 or 2
                continue
            else:
                image, origin, spacing, direction, image_type = get_medical_image(os.path.join(path, file.name))
                image = np.rot90(image, k=-2, axes=(0, 1))
                save_medical_image(image, os.path.join(save_path, file.name), origin=origin, spacing=spacing, direction=direction, type=image_type)


def get_k_folder(image_dire, dst_file='./k-folder.json', K=None):
    """
    生成K折交叉验证
    :param image_dire:
    :param dst_file:
    :param K:
    :return:
    """
    files = get_files_name(dire=image_dire)
    results = get_k_folder_cross_validation_samples(files, K)
    save_json(results, file=dst_file)
