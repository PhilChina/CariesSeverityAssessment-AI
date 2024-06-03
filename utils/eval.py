
import math
import numpy as np


## dice
from utils.pre_processing import norm_zero_one


def get_dice(result, reference):
    '''
    计算两个二值数组之间的dice系数
    math::
    $$ DICE=\frac{2TP}{2TP+FP+FN}=\frac{2|A\cap B|}{|A|+|B|} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: dice [0,1] 0-完全无重叠部分 1-完全重叠在一起
    '''
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


## jaccard
def get_jac(result, reference):
    '''
    计算两个二值数组之间的jaccard系数
    math::
    $$ Jaccard=\frac{|A\cap B|}{|A\cup B|}=\frac{DICE}{2-DICE} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: jaccard [0,1] 0-完全无重叠部分 1-完全重叠在一起
    '''
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)

    try:
        jaccard = float(intersection) / float(union)
    except ZeroDivisionError:
        jaccard = 0.0

    return jaccard


## true positive rate
def get_tpr(result, reference):
    '''
    真阳率：将正类预测为正类的比例
    True positive rate
    math::
    $$ TPR = \frac{TP}{TP+FN} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: true positive rate
    '''
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    true_positive_rate = float(tp) / (float(tp) + float(fn))

    return true_positive_rate


## true negative rate
def get_tnr(result, reference):
    '''
    真阴率：将负类预测为负类的比例
    True positive rate
    math::
    $$ TNR = \frac{TN}{TN+FP} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: true negative rate
    '''
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)

    true_negative_rate = float(tn) / (float(tn) + float(fp))
    return true_negative_rate


## false positive rate
def get_fpr(result, reference):
    return 1. - get_tnr(result, reference)


## false negative rate
def get_fnr(result, reference):
    return 1. - get_tpr(result, reference)


## recall
def get_recall(result, reference):
    '''
    召回率: 覆盖面的度量 - 度量多少个正例被正确分类
    recall
    math::
    $$ Recall = \frac{TP}{TP+FN} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: true positive rate
    '''
    return get_tpr(result, reference)


## sensitivity
def get_sensitivity(result, reference):
    '''
    敏感度: 衡量了分类器对正类的识别能力
    recall
    math::
    $$ Recall = \frac{TP}{TP+FN} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: true positive rate
    '''
    return get_tpr(result, reference)


## specificity
def get_specificity(result, reference):
    '''
    特异性: 衡量分类器对负类的识别能力
    specificity
    math::
    $$ specificity = \frac{TN}{TN+FP} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: true negative rate
    '''
    return get_tnr(result, reference)


## precision
def get_precision(result, reference):
    '''
    精确率，精度：衡量被分类为正例中真实为正例的比例
    precision
    math::
    $$ Precision = \frac{TP}{TP+FP} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: precision
    '''
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)

    precision = float(tp) / (float(tp) + float(fp))

    return precision


## accuracy
def get_accuracy(result, reference):
    '''
    准确率：分对的数目/所有数目的比
    accuracy
    math::
    $$ Accuracy = \frac{TP+TN}{TP+TN+FP+FN} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: accuracy
    '''
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)
    fn = np.count_nonzero(~result & reference)

    accuracy = (float(tp) + float(tn)) / (float(tp) + float(tn) + float(fp) + float(fn))
    return accuracy


## fall out
def get_fallout(result, reference):
    return 1.0 - get_tnr(result, reference)


## global consistency error
def get_gce(result, reference):
    '''
    Global Consistency Error
    $R(S,x)$ is defined as the set of all voxels that
    reside in the same region of segmentation $S$
    where the voxel $x$ resides. [Taha,2015]
    $E(S_{t},S_{g},x)=\frac{|R(S_{T},x)/R(S_{g},x)|}{R(S_{T},x)}$
    $$ GCE(S_{t},S_{g})=\frac{1}{n}\min\{\sum \limits_{i}^{n}(S_{t},S_{g},x_{i}),\sum \limits_{i}^{n}(S_{g},S_{t},x_{i})\} $$
    $$ GCE(S_{t},S_{g})=\frac{1}{n}\min\{\frac{FN(FN+2TP)}{TP+FN}+\frac{FP(FP+2TN)}{TN+FP},\frac{FP(FP+2TP)}{TP+FP}+\frac{FN(FN+2TN)}{TN+FN}\} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: global consistency error
    '''
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = float(np.count_nonzero(result & reference))
    tn = float(np.count_nonzero(~result & ~reference))
    fp = float(np.count_nonzero(result & ~reference))
    fn = float(np.count_nonzero(~result & reference))

    n = tp + tn + fp + fn

    e1 = (fn * (fn + 2 * tp) / (tp + fn) + fp * (fp + 2 * tn) / (tn + fp)) / n
    e2 = (fp * (fp + 2 * tp) / (tp + fp) + fn * (fn + 2 * tn) / (tn + fn)) / n

    return min(e1, e2)


## volumetric similarity
def get_vs(result, reference):
    '''
    Volumetric Similarity
    math:
    $$ VS = 1-\frac{||A|-|B||}{|A|+|B|}=1-\frac{|FN-FP|}{2TP+FP+FN} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: global consistency error
    '''

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        vs = 1 - (abs(fn - fp) / float(2 * tp + fp + fn))
    except ZeroDivisionError:
        vs = 0.0

    return vs


## Rand Index
def get_ri(result, reference):
    '''
    Rand Index
    math:
    $$
    \begin{cases}
        & a = \frac{1}{2}[TP(TP-1)+FP(FP-1)+TN(TN-1)+FN(FN-1)] \\
        & b = \frac{1}{2}[(TP+FN)^{2}+(TN+FP)^{2}-(TP^{2}+TN^{2}+FP^{2}+FN^{2})] \\
        & c = \frac{1}{2}[(TP+FP)^{2}+(TN+FN)^{2}-(TP^{2}+TN^{2}+FP^{2}+FN^{2})] \\
        & d = n(n-1)/2-(a+b+c)
    \end{cases}
    $$
    $$ RI(A,B) = \frac{a+b}{a+b+c+d} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: rand index
    '''

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = float(np.count_nonzero(result & reference))
    tn = float(np.count_nonzero(~result & ~reference))
    fp = float(np.count_nonzero(result & ~reference))
    fn = float(np.count_nonzero(~result & reference))

    n = tp + tn + fp + fn

    a = (tp * (tp - 1) + fp * (fp - 1) + tn * (tn - 1) + fn * (fn - 1)) / 2
    b = ((tp + fn) ** 2 + (tn + fp) ** 2 - (tp ** 2 + tn ** 2 + fp ** 2 + fn ** 2)) / 2
    c = ((tp + fp) ** 2 + (tn + fn) ** 2 - (tp ** 2 + tn ** 2 + fp ** 2 + fn ** 2)) / 2
    d = n * (n - 1) / 2 - (a + b + c)

    ri = (a + b) / (a + b + c + d)

    return ri


## Adjusted Rand Index
def get_ari(result, reference):
    '''
    Adjusted Rand Index
    math:
    $$
    \begin{cases}
        & a = \frac{1}{2}[TP(TP-1)+FP(FP-1)+TN(TN-1)+FN(FN-1)] \\
        & b = \frac{1}{2}[(TP+FN)^{2}+(TN+FP)^{2}-(TP^{2}+TN^{2}+FP^{2}+FN^{2})] \\
        & c = \frac{1}{2}[(TP+FP)^{2}+(TN+FN)^{2}-(TP^{2}+TN^{2}+FP^{2}+FN^{2})] \\
        & d = n(n-1)/2-(a+b+c)
    \end{cases}
    $$
    $$ ARI(A,B) = \frac{2(ad-bc)}{c^{2}+b^{2}+2ad+(a+d)(c+b)} $$
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: adjusted rand index
    '''

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = float(np.count_nonzero(result & reference))
    tn = float(np.count_nonzero(~result & ~reference))
    fp = float(np.count_nonzero(result & ~reference))
    fn = float(np.count_nonzero(~result & reference))

    n = tp + tn + fp + fn

    a = (tp * (tp - 1) + fp * (fp - 1) + tn * (tn - 1) + fn * (fn - 1)) / 2
    b = ((tp + fn) ** 2 + (tn + fp) ** 2 - (tp ** 2 + tn ** 2 + fp ** 2 + fn ** 2)) / 2
    c = ((tp + fp) ** 2 + (tn + fn) ** 2 - (tp ** 2 + tn ** 2 + fp ** 2 + fn ** 2)) / 2
    d = n * (n - 1) / 2 - (a + b + c)

    ari = (2 * (a * d - b * c)) / (c ** 2 + b ** 2 + 2 * a * d + (a + d) * (c + b))
    return ari


## Mutual Information
def get_mi(result, reference):
    '''
    Mutual Information
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: adjusted rand index
    '''

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = float(np.count_nonzero(result & reference))
    tn = float(np.count_nonzero(~result & ~reference))
    fp = float(np.count_nonzero(result & ~reference))
    fn = float(np.count_nonzero(~result & reference))

    n = tp + tn + fp + fn

    row1 = tn + fn
    row2 = fp + tp
    H1 = - ((row1 / n) * math.log(row1 / n, 2) + (row2 / n) * math.log(row2 / n, 2))

    col1 = tn + fp
    col2 = fn + tp
    H2 = - ((col1 / n) * math.log(col1 / n, 2) + (col2 / n) * math.log(col2 / n, 2))

    p00 = 1 if tn == 0 else tn / n
    p01 = 1 if fn == 0 else fn / n
    p10 = 1 if fp == 0 else fp / n
    p11 = 1 if tp == 0 else tp / n
    H12 = - ((tn / n) * math.log(p00, 2) +
             (fn / n) * math.log(p01, 2) +
             (fp / n) * math.log(p10, 2) +
             (tp / n) * math.log(p11, 2))
    MI = H1 + H2 - H12
    return MI


## Variation of Information
def get_voi(result, reference):
    '''
    Variation Information
    :param result: 可以二值化的数组，背景为0，前景为非0
    :param reference: 可以二值化的数组，背景为0，前景为非0
    :return: adjusted rand index
    '''

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = float(np.count_nonzero(result & reference))
    tn = float(np.count_nonzero(~result & ~reference))
    fp = float(np.count_nonzero(result & ~reference))
    fn = float(np.count_nonzero(~result & reference))

    n = tp + tn + fp + fn

    row1 = tn + fn
    row2 = fp + tp
    H1 = - ((row1 / n) * math.log(row1 / n, 2) + (row2 / n) * math.log(row2 / n, 2))

    col1 = tn + fp
    col2 = fn + tp
    H2 = - ((col1 / n) * math.log(col1 / n, 2) + (col2 / n) * math.log(col2 / n, 2))

    p00 = 1 if tn == 0 else tn / n
    p01 = 1 if fn == 0 else fn / n
    p10 = 1 if fp == 0 else fp / n
    p11 = 1 if tp == 0 else tp / n
    H12 = - ((tn / n) * math.log(p00, 2) +
             (fn / n) * math.log(p01, 2) +
             (fp / n) * math.log(p10, 2) +
             (tp / n) * math.log(p11, 2))
    MI = H1 + H2 - H12

    VOI = H1 + H2 - 2 * MI

    return VOI


## Interclass correlation
def get_icc(result, reference):
    result = np.ravel(result.astype(np.float))
    reference = np.ravel(reference.astype(np.float))

    mean_result = np.mean(result)
    mean_reference = np.mean(reference)

    assert reference.shape[0] == result.shape[0], "shape does't match"
    n = reference.shape[0]

    ssw = 0.
    ssb = 0.
    gradmean = (mean_result + mean_reference) / 2.
    for i in range(n):
        r1 = result[i]
        r2 = reference[i]
        m = (r1 + r2) / 2

        ssw += pow(r1 - m, 2)
        ssw += pow(r2 - m, 2)
        ssb += pow(m - gradmean, 2)

    ssw = ssw / n
    ssb = ssb / (n - 1) * 2

    icc = (ssb - ssw) / (ssb + ssw)
    return icc


## Probability distance
def get_pbd(result, reference):
    # result = np.atleast_1d(result.astype(np.bool))
    # reference = np.atleast_1d(reference.astype(np.bool))
    result = np.ravel(result.astype(np.float))
    reference = np.ravel(reference.astype(np.float))

    assert reference.shape[0] == result.shape[0], "shape does't match"
    n = reference.shape[0]

    probability_joint = 0.
    probability_diff = 0.

    for i in range(n):
        r1 = result[i]
        r2 = reference[i]

        probability_diff += abs(r1 - r2)
        probability_joint += r1 * r2

    pd = -1
    if probability_joint != 0:
        pd = probability_diff / (2 * probability_joint)

    return pd


## Cohen Kappa Coefficient
def get_kap(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = float(np.count_nonzero(result & reference))
    tn = float(np.count_nonzero(~result & ~reference))
    fp = float(np.count_nonzero(result & ~reference))
    fn = float(np.count_nonzero(~result & reference))

    agreement = tp + tn

    chance_0 = (tn + fn) * (tn + fp)
    chance_1 = (fp + tp) * (fn + tp)
    chance = chance_0 + chance_1

    sum = (tn + fn + fp + tp)
    chance = chance / sum

    kappa = (agreement - chance) / (sum - chance)
    return kappa


## Area under ROC curve
def get_auc(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = float(np.count_nonzero(result & reference))
    tn = float(np.count_nonzero(~result & ~reference))
    fp = float(np.count_nonzero(result & ~reference))
    fn = float(np.count_nonzero(~result & reference))

    auc = 1 - (fp / (fp + tn) + fn / (fn + tp)) / 2
    return auc


## Hausdorf distance
def get_hd(result, reference):
    from medpy.metric.binary import hd
    return hd(result, reference)


## Average surface distance
def get_asd(result, reference):
    from medpy.metric.binary import asd
    return asd(result, reference)


## Average symmetric surface distance.
def get_assd(result, reference):
    from medpy.metric.binary import assd
    return assd(result, reference)


## Mahalanobis distance
# def mhd(result,reference):

## peak signal noise ratio
def get_psnr(result, reference,data_range=None):
    from skimage.metrics import peak_signal_noise_ratio
    return peak_signal_noise_ratio(result, reference,data_range=data_range)


## mean squared error
def get_mse(result, reference):
    from skimage.metrics import mean_squared_error
    return mean_squared_error(result, reference)


## normalized root mean squared error
def get_nrmse(result, reference):
    from skimage.metrics import normalized_root_mse
    return normalized_root_mse(result, reference)


## structural similarity
def get_ssim(result, reference, data_range=None,win_size=None):
    from skimage.metrics import structural_similarity
    return structural_similarity(result, reference, data_range=data_range,win_size=win_size)


## get kl divergence
def get_kl_div(pk, qk):
    '''
    D_{KL} (p_{k}||q_{k})
    :param pk:
    :param qk:
    Attention: 不能有0
    :return:
    '''
    from scipy.stats import entropy

    pk = norm_zero_one(pk) * 0.98 + 0.01
    qk = norm_zero_one(qk) * 0.98 + 0.01

    pk = np.asarray(pk).flatten()
    qk = np.asarray(qk).flatten()

    pk = pk / np.sum(pk)
    qk = qk / np.sum(qk)

    kl = entropy(pk, qk)
    return kl


def all(result, reference):
    metric = {
        "dice: ": get_dice(result, reference),
        "jaccard: ": get_jac(result, reference),
        "true positive rate: ": get_tpr(result, reference),
        "true negative rate: ": get_tnr(result, reference),
        "false positive rate: ": get_fpr(result, reference),
        "false negative rate: ": get_fnr(result, reference),
        "recall: ": get_recall(result, reference),
        "sensitivity: ": get_sensitivity(result, reference),
        "specificity: ": get_specificity(result, reference),
        "precision: ": get_precision(result, reference),
        "accuracy: ": get_accuracy(result, reference),
        "fall out: ": get_fallout(result, reference),
        "global consistency error: ": get_gce(result, reference),
        "volumetric similarity: ": get_vs(result, reference),
        "rand index: ": get_ri(result, reference),
        "adjusted rand index: ": get_ari(result, reference),
        "mutual information: ": get_mi(result, reference),
        "variation of information: ": get_voi(result, reference),
        "interclass correlation: ": get_icc(result, reference),
        "probability distance: ": get_pbd(result, reference),
        "cohen kappa distance: ": get_kap(result, reference),
        "area under ROC curve: ": get_auc(result, reference),
        "hausdorff distance: ": get_hd(result, reference),
        "average surface distance: ": get_asd(result, reference),
        "average symmetric surface distance: ": get_assd(result, reference)
    }
    return metric
