import collections
import csv
import inspect
import json
import os
import re

import numpy as np
import torch
from PIL import Image
from skimage.exposure import rescale_intensity
## 得到K个不相交的子集
from skimage.measure import label
from skimage.morphology import convex_hull_object


def get_k_samples(total_num, K):
    """
    K折验证随机分成K个
    :param total_num: 样本总数
    :param K: 折数
    :return: [ [] ...] 总共有K个list
    """
    import random
    num = total_num // K

    ## 所有的样本
    samples = set(range(total_num))

    result = []
    for index in range(K - 1):
        if len(samples) < num:
            break
        tmp = set(random.sample(samples, num))
        result.append(tmp)
        samples = samples - tmp
    result.append(samples)

    # from pprint import pprint
    # pprint(result)
    return result


def get_k_samples_in_order(order, K):
    import numpy as np
    '''
    :param order: index in order
    :param K:
    :return:
    '''
    ## total number of samples
    num = len(order)

    ## index of arrangement
    order = np.asarray(order)
    index = np.argsort(order)

    ## result
    arrangements = []
    for i in range(num // K):
        arrangements.append(list(index[i * K:(i + 1) * K if i != num // K - 1 else num - 1]))

    results = []
    for k in range(K - 1):
        tmp_result = []
        for i in range(num // K):
            tmp_index = np.random.choice(arrangements[i])
            tmp_result.append(tmp_index)
            del (arrangements[i][arrangements[i].index(tmp_index)])
        results.append(tmp_result)

    arrangements = list(eval(str(arrangements).replace('[', '').replace(']', '')))
    results.append(arrangements)
    return results


## 得到K折验证的子集
def get_k_folder_cross_validation_samples(files, K, order=None):
    """
    :param files: Samples file , such as dicom
    :param K:     K folder cross validation
    :param order: if not None arrangement in specific random order else random order (default: None)
    :return:
    """
    total_num = len(files)
    if order is None:
        samples = get_k_samples(total_num=total_num, K=K)
    else:
        assert len(order) == total_num, "order num doesn't match the file numer"
        samples = get_k_samples_in_order(order, K)

    result = dict()
    for index in range(K):
        test = [files[s] for s in samples[index]]
        train = list(set(files) - set(test))
        result[index] = {"train": train, "test": test}

    return result

def get_large_region(image):
    ## 3D
    result = []
    for slice in image:
        labels = label(input=slice, background=0, connectivity=2, return_num=False)
        largestCC = labels == np.argmax(np.bincount(labels.flat, weights=slice.flat))
        if np.sum(slice):
            result.append(largestCC)
        else:
            result.append(np.zeros(slice.shape))
    return np.asarray(result).astype(int)

def get_2d_image_large_region(image):
    labels = label(input=image, background=0, connectivity=2, return_num=False)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=image.flat))
    if np.sum(image):
        return np.asarray(largestCC).astype(int)
    else:
        return np.zeros(image.shape)


def hull_image(image):
    result = []
    for slice in image:
        if np.sum(slice):
            # slice = convex_hull_image(slice, tolerance=10.0)
            slice = convex_hull_object(slice)
            result.append(slice)
        else:
            result.append(np.zeros(slice.shape))
    return np.asarray(result).astype(int)



## 获得标签的最大的N个连通区域
def get_largest_n_connected_region(mask, n, background=0):
    """
    将得到的标签求最大连通区域
    :param n:
    :param mask: 得到的标签数组
    :param background: 标签数组的背景，默认是0
    :return: largest_connected_region，numpy数组，只包含0,1标签
    因为用到skimage中的函数label，所以这里以mask指代标签
    """
    from skimage.measure import label
    import numpy as np

    ## 返回每个连通区域，将每个连通区域赋予不同的像素值
    mask = label(mask, background=background)
    mask_flat = mask.flat

    ## bincount 标签中每个索引值的个数
    ## E.g. Array: [0, 1, 1, 3, 2, 1, 7]
    ## 遍历 0→7的索引，得到索引值得个数：[1, 3, 1, 1, 0, 0, 0, 1]，索引0出现了1次，索引1出现了3次...索引7出现了1次
    index_num = np.bincount(mask_flat)

    ## 将像素值出现的次数进行排序[从大到小]，选出非背景的像素值最多（最大连通）的像素值索引
    ## 而排序时产生的index其实就是像素值
    pixel_index = np.argsort(-index_num)
    pixel = pixel_index[0]

    connected_area = []
    for p in pixel_index:
        if p != background:
            tmp_area = np.zeros(mask.shape)
            tmp_area[mask == p] = 1
            tmp_area = np.asarray(tmp_area).astype(np.int)
            connected_area.append(tmp_area)
            if len(connected_area) == n:
                break

    return np.asarray(connected_area)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imgtype='img', datatype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.ndim == 4:  # image_numpy (C x W x H x S)
        mid_slice = image_numpy.shape[-1]//2
        image_numpy = image_numpy[:,:,:,mid_slice]
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if imgtype == 'img':
        image_numpy = (image_numpy + 8) / 16.0 * 255.0
    if np.unique(image_numpy).size == int(1):
        return image_numpy.astype(datatype)
    return rescale_intensity(image_numpy.astype(datatype))


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())


def determine_crop_size(inp_shape, div_factor):
    div_factor= np.array(div_factor, dtype=np.float32)
    new_shape = np.ceil(np.divide(inp_shape, div_factor)) * div_factor
    pre_pad = np.round((new_shape - inp_shape) / 2.0).astype(np.int16)
    post_pad = ((new_shape - inp_shape) - pre_pad).astype(np.int16)
    return pre_pad, post_pad


def csv_write(out_filename, in_header_list, in_val_list):
    with open(out_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(in_header_list)
        writer.writerows(zip(*in_val_list))
