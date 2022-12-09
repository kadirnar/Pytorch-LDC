import os

import cv2
import kornia as kn
import numpy as np
import torch

from utils.config_utils import get_config


def stack(*args):
    return np.hstack(args)


def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def count_parameters(model=None):
    if model is not None:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        print("Error counting model parameters line 32 img_processing.py")
        raise NotImplementedError


def save_image_batch_to_disk(
    tensor,
    output_dir,
    file_names,
    img_shape=None,
    is_testing=False,
    is_inchannel=False,
    config_path="data/config/default_config.yaml",
):
    os.makedirs(output_dir, exist_ok=True)
    if not is_testing:
        assert len(tensor.shape) == 4, tensor.shape
        img_shape = np.array(img_shape)
        for tensor_image, file_name in zip(tensor, file_names):
            image_vis = kn.utils.tensor_to_image(torch.sigmoid(tensor_image))  # [..., 0]
            image_vis = (255.0 * (1.0 - image_vis)).astype(np.uint8)
            output_file_name = os.path.join(output_dir, file_name)
            image_vis = cv2.resize(image_vis, dsize=(int(img_shape[1]), int(img_shape[0])))
            assert cv2.imwrite(output_file_name, image_vis)
    else:
        tensor2 = None
        tmp_img2 = None
        edge_maps = []
        for i in tensor:
            tmp = torch.sigmoid(i).cpu().detach().numpy()
            edge_maps.append(tmp)
        tensor = np.array(edge_maps)
        # breakpoint()
        image_shape = [x.cpu().detach().numpy() for x in img_shape]
        # (H, W) -> (W, H)
        image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

        assert len(image_shape) == len(file_names)

        idx = 0
        for i_shape, file_name in zip(image_shape, file_names):
            tmp = tensor[:, idx, ...]
            tmp2 = tensor2[:, idx, ...] if tensor2 is not None else None
            # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
            tmp = np.squeeze(tmp)
            tmp2 = np.squeeze(tmp2) if tensor2 is not None else None

            # Iterate our all 7 NN outputs for a particular image
            preds = []
            fuse_num = tmp.shape[0] - 1

            for i in range(tmp.shape[0]):
                tmp_img = tmp[i]
                tmp_img = np.uint8(image_normalization(tmp_img))
                tmp_img = cv2.bitwise_not(tmp_img)

                # Resize prediction to match input image size
                if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                    tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
                    tmp_img2 = cv2.resize(tmp_img2, (i_shape[0], i_shape[1])) if tmp2 is not None else None

                else:
                    preds.append(tmp_img)

                if i == fuse_num:
                    # print('fuse num',tmp.shape[0], fuse_num, i)
                    fuse = tmp_img
                    fuse = fuse.astype(np.uint8)

            # Get the mean prediction of all the 7 outputs
            config = get_config(config_path)
            model_path = config.TRAIN_CONFIG.CHECKPOINT_DATA
            model_name = model_path.split("/")[-1]
            model_name = model_name.split(".")[0]

            save_image_path = os.path.join(output_dir, model_name)
            os.makedirs(save_image_path, exist_ok=True)
            average = np.array(preds, dtype=np.float32)
            average = np.uint8(np.mean(average, axis=0))
            output_file_name = os.path.join(save_image_path, file_name)
            # multi_img = stack(fuse, average)
            cv2.imwrite(output_file_name, fuse)

            idx += 1
            return fuse


def restore_rgb(config, I, restore_rgb=False):
    """
    :param config: [args.channel_swap, args.mean_pixel_value]
    :param I: and image or a set of images
    :return: an image or a set of images restored
    """

    if len(I) > 3 and not type(I) == np.ndarray:
        I = np.array(I)
        I = I[:, :, :, 0:3]
        n = I.shape[0]
        for i in range(n):
            x = I[i, ...]
            x = np.array(x, dtype=np.float32)
            x += config[1]
            if restore_rgb:
                x = x[:, :, config[0]]
            x = image_normalization(x)
            I[i, :, :, :] = x
    elif len(I.shape) == 3 and I.shape[-1] == 3:
        I = np.array(I, dtype=np.float32)
        I += config[1]
        if restore_rgb:
            I = I[:, :, config[0]]
        I = image_normalization(I)
    else:
        print("Sorry the input data size is out of our configuration")
    return I


def visualize_result(imgs_list):
    """
    data 2 image in one matrix
    :param imgs_list: a list of prediction, gt and input data
    :param arg:
    :return: one image with the whole of imgs_list data
    """
    mean_pixel_values = [103.939, 116.779, 123.68, 137.86]
    channel_swap = [2, 1, 0]
    n_imgs = len(imgs_list)
    data_list = []
    for i in range(n_imgs):
        tmp = imgs_list[i]
        # print(tmp.shape)
        if tmp.shape[0] == 3:
            tmp = np.transpose(tmp, [1, 2, 0])
            tmp = restore_rgb([channel_swap, mean_pixel_values[:3]], tmp)
            tmp = np.uint8(image_normalization(tmp))
        else:
            tmp = np.squeeze(tmp)
            if len(tmp.shape) == 2:
                tmp = np.uint8(image_normalization(tmp))
                tmp = cv2.bitwise_not(tmp)
                tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
            else:
                tmp = np.uint8(image_normalization(tmp))
        data_list.append(tmp)
        # print(i,tmp.shape)
    img = data_list[0]
    if n_imgs % 2 == 0:
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1] * (n_imgs // 2) + ((n_imgs // 2 - 1) * 5), 3))
    else:
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1] * ((1 + n_imgs) // 2) + ((n_imgs // 2) * 5), 3))
        n_imgs += 1

    k = 0
    imgs = np.uint8(imgs)
    i_step = img.shape[0] + 10
    j_step = img.shape[1] + 5
    for i in range(2):
        for j in range(n_imgs // 2):
            if k < len(data_list):
                imgs[i * i_step : i * i_step + img.shape[0], j * j_step : j * j_step + img.shape[1], :] = data_list[k]
                k += 1
            else:
                pass
    return imgs
