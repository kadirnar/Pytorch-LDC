import math

import cv2
import numpy as np


def calculate_precision(y_true, y_pred, threshold=0.1):
    """
    Calculates the precision metric for a binary classification problem.

    Args:
        y_true: A numpy array of true labels.
        y_pred: A numpy array of predicted labels.
        threshold: A float value representing the classification threshold.

    Returns:
        A float value representing the precision.
    """
    # Calculate true positive
    tp = np.sum((y_true == 1) & (y_pred > threshold))

    # Calculate false positive
    fp = np.sum((y_true == 0) & (y_pred > threshold))

    # Calculate precision
    precision = tp / (tp + fp)

    return precision


def calculate_recall(y_true, y_pred, threshold=0.1):
    """
    Calculates the recall metric for a binary classification problem.

    Args:
        y_true: A numpy array of true labels.
        y_pred: A numpy array of predicted labels.
        threshold: A float value representing the classification threshold.

    Returns:
        A float value representing the recall.
    """

    # Calculate true positive
    tp = np.sum((y_true == 1) & (y_pred > threshold))

    # Calculate false positive
    fp = np.sum((y_true == 0) & (y_pred > threshold))

    # Calculate false negative
    fn = np.sum((y_true == 1) & (y_pred <= threshold))

    # Calculate recall
    recall = tp / (tp + fn)

    return recall


def calculate_ods_f_score(y_true, y_pred, beta=1.0):
    """
    Calculates the ODS-F-Score metric for a binary classification problem.

    Args:
        y_true: A numpy array of true labels.
        y_pred: A numpy array of predicted labels.
        beta: A float value representing the weight of precision in the metric.

    Returns:
        A float value representing the ODS-F-Score.
    """

    # Calculate true positive
    tp = np.sum((y_true == 1) & (y_pred > 0.5))

    # Calculate false positive
    fp = np.sum((y_true == 0) & (y_pred > 0.5))

    # Calculate false negative
    fn = np.sum((y_true == 1) & (y_pred <= 0.5))

    # Calculate precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # Calculate ODS-F-Score
    ods_f_score = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)

    return ods_f_score


def psnr(original_path, contrast_path):
    original = cv2.imread(original_path)
    contrast = cv2.imread(contrast_path)
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR


def ssim(original_path, contrast_path):
    from skimage.metrics import structural_similarity

    # İki görüntüyü oku
    image1 = cv2.imread(original_path)
    image2 = cv2.imread(contrast_path)

    # Görüntüleri gri tonlamaya çevir
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # SSIM'i hesapla
    (score, diff) = structural_similarity(image1, image2, full=True)
    return score
