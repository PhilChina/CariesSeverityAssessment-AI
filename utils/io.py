import os
import natsort
import SimpleITK as sitk
from PIL import Image
import numpy as np


## 按顺序得到当前目录下，所有文件（包括文件夹）的名字
def get_files_name(dire):
    """
    按顺序得到当前目录下，所有文件（包括文件夹）的名字
    :param dire: 文件夹目录
    :return:files[list]，当前目录下所有的文件（包括文件夹）的名字，顺序排列
    """

    assert os.path.exists(dire), "{} is not existed".format(dire)
    assert os.path.isdir(dire), "{} is not a directory".format(dire)

    files = os.listdir(dire)
    files = natsort.natsorted(files)
    return files


## 得到2D/3D的医学图像(除.dcm序列图像)
def get_medical_image(path):
    """
    加载一幅2D/3D医学图像(除.dcm序列图像)，支持格式：.nii, .nrrd, ...
    :param path: 医学图像的路径/SimpleITK.SimpleITK.Image
    :return:(array,origin,spacing,direction)
    array:  图像数组
    origin: 三维图像坐标原点
    spacing: 三维图像坐标间距
    direction: 三维图像坐标方向
    image_type: 图像像素的类型
    注意：实际取出的数组不一定与MITK或其他可视化工具中的方向一致！
    可能会出现旋转、翻转等现象，这是由于dicom头文件中的origin,spacing,direction的信息导致的
    在使用时建议先用matplotlib.pyplot工具查看一下切片的方式是否异常，判断是否需要一定的预处理
    """

    if isinstance(path, sitk.Image):
        reader = path
    else:
        assert os.path.exists(path), "{} is not existed".format(path)
        assert os.path.isfile(path), "{} is not a file".format(path)
        reader = sitk.ReadImage(path)

    array = sitk.GetArrayFromImage(reader)
    spacing = reader.GetSpacing()  ## 间隔
    origin = reader.GetOrigin()  ## 原点
    direction = reader.GetDirection()  ## 方向
    image_type = reader.GetPixelID()  ## 原图像每一个像素的类型，
    return array, origin, spacing, direction, image_type


## 将numpy数组保存为3D医学图像格式，支持 .nii, .nrrd
def save_medical_image(array, target_path, origin=None, spacing=None, direction=None, type=None):
    """
    将得到的数组保存为医学图像格式
    :param array: 想要保存的医学图像数组，为避免错误，这个函数只识别3D数组
    :param origin:读取原始数据中的原点
    :param spacing: 读取原始数据中的间隔
    :param direction: 读取原始数据中的方向
    :param target_path: 保存的文件路径，注意：一定要带后缀，E.g.,.nii,.nrrd SimpleITK会根据路径的后缀自动判断格式，填充相应信息
    :param type: 像素的储存格式
    :return: None 无返回值
    注意，因为MITK中会自动识别当前载入的医学图像文件是不是标签(label)【通过是否只有0,1两个值来判断】
    所以在导入的时候，MITK会要求label的文件格式为unsigned_short/unsigned_char型，否则会有warning
    """

    assert len(np.asarray(array).shape) == 3, "array's shape is {}, it's not a 3D array".format(np.asarray(array).shape)

    ## if isVector is true, then a 3D array will be treaded as a 2D vector image
    ## otherwise it will be treaded as a 3D image
    image = sitk.GetImageFromArray(array, isVector=False)
    if direction is not None: image.SetDirection(direction)
    if spacing is not None: image.SetSpacing(spacing)
    if origin is not None: image.SetOrigin(origin)

    if type is None:
        sitk.WriteImage(sitk.Cast(image, sitk.sitkInt32), target_path, True)
    else:
        ## 如果是标签，按照MITK要求改为unsigned_char/unsigned_short型 [sitk.sitkUInt8]
        sitk.WriteImage(sitk.Cast(image, type), target_path, True)


## 加载一张普通格式图片 2D
def get_normal_image(path):
    """
    加载一幅普通格式的2D图像，支持格式：.jpg, .jpeg, .tif ...
    :param path: 医学图像的路径
    :return: array: numpy格式
    """
    array = Image.open(path)
    array = np.asarray(array)
    return array


## get json content as dict
def get_json(file):
    import json
    with open(file, 'r', encoding='utf-8') as f:
        dicts = json.load(f)
    return dicts


## save dict as json
def save_json(dicts, file, indent=2):
    import json
    info = json.dumps(dicts, indent=indent, ensure_ascii=False)
    with open(file, 'w', encoding='utf-8') as f:  # 使用.dumps()方法时，要写入
        f.write(info)


## save obj as pickle
def save_pickle(obj, file):
    import pickle
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)