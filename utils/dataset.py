import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, data_root, test_data, mean_bgr, img_height, img_width, test_list=None):
        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        if not self.test_list:
            raise ValueError(f"Test list not provided for dataset: {self.test_data}")

        list_name = os.path.join(self.data_root, self.test_list)
        if self.test_data.upper() in ["BIPED", "BRIND"]:
            # breakpoint()
            with open(list_name) as f:
                files = json.load(f)
            for pair in files:
                tmp_img = pair[0]
                tmp_gt = pair[1]
                sample_indices.append(
                    (
                        os.path.join(self.data_root, tmp_img),
                        os.path.join(self.data_root, tmp_gt),
                    )
                )
        else:
            with open(list_name, "r") as f:
                files = f.readlines()
            files = [line.strip() for line in files]
            pairs = [line.split() for line in files]

            for pair in pairs:
                tmp_img = pair[0]
                tmp_gt = pair[1]
                sample_indices.append(
                    (
                        os.path.join(self.data_root, tmp_img),
                        os.path.join(self.data_root, tmp_gt),
                    )
                )
        return sample_indices

    def __len__(self):
        return len(self.data_index[0]) if self.test_data.upper() == "CLASSIC" else len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx] if len(self.data_index[0]) > 1 else self.data_index[0][idx - 1]
        else:
            image_path = self.data_index[idx][0]
        label_path = None if self.test_data == "CLASSIC" else self.data_index[idx][1]
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        # base dir
        if self.test_data.upper() == "BIPED":
            img_dir = os.path.join(self.data_root, "imgs", "test")
            gt_dir = os.path.join(self.data_root, "edge_maps", "test")
        elif self.test_data.upper() == "CLASSIC":
            img_dir = self.data_root
            gt_dir = None
        else:
            img_dir = self.data_root
            gt_dir = self.data_root

        # load data
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        if not self.test_data == "CLASSIC":
            label = cv2.imread(os.path.join(gt_dir, label_path), cv2.IMREAD_COLOR)
        else:
            label = None

        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        if self.test_data == "CLASSIC":
            img_height = self.img_height
            img_width = self.img_width
            print(f"actual size: {img.shape}, target size: {(img_height, img_width,)}")
            img = cv2.resize(img, (img_width, img_height))
            gt = None

        # Make images and labels at least 512 by 512
        elif img.shape[0] < 512 or img.shape[1] < 512:
            img = cv2.resize(img, (self.img_width, self.img_height))
            gt = cv2.resize(gt, (self.img_width, self.img_height))

        # Make sure images and labels are divisible by 2^4=16
        elif img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
            img_width = ((img.shape[1] // 16) + 1) * 16
            img_height = ((img.shape[0] // 16) + 1) * 16
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))
        else:
            img_width = self.img_width
            img_height = self.img_height
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.0
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


class BipedDataset(Dataset):
    train_modes = [
        "train",
        "test",
    ]
    dataset_types = [
        "rgbr",
    ]
    data_types = [
        "aug",
    ]

    def __init__(
        self,
        data_root,
        img_height,
        img_width,
        mean_bgr,
        train_mode="train",
        dataset_type="rgbr",
        train_data="BIPED",
        train_list="",
        crop_img=False,
    ):
        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = "aug"  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.train_data = train_data
        self.train_list = train_list

        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []
        if self.train_data.lower() == "biped":
            images_path = os.path.join(data_root, "edges/imgs", self.train_mode, self.dataset_type, self.data_type)
            labels_path = os.path.join(data_root, "edges/edge_maps", self.train_mode, self.dataset_type, self.data_type)

            for directory_name in os.listdir(images_path):
                image_directories = os.path.join(images_path, directory_name)
                for file_name_ext in os.listdir(image_directories):
                    file_name = os.path.splitext(file_name_ext)[0]
                    sample_indices.append(
                        (
                            os.path.join(images_path, directory_name, file_name + ".jpg"),
                            os.path.join(labels_path, directory_name, file_name + ".png"),
                        )
                    )
        else:
            file_path = os.path.join(data_root, self.train_list)
            if self.train_data.lower() == "bsds":

                with open(file_path, "r") as f:
                    files = f.readlines()
                files = [line.strip() for line in files]

                pairs = [line.split() for line in files]
                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (
                            os.path.join(data_root, tmp_img),
                            os.path.join(data_root, tmp_gt),
                        )
                    )
            else:
                with open(file_path) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (
                            os.path.join(data_root, tmp_img),
                            os.path.join(data_root, tmp_gt),
                        )
                    )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255.0

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        i_h, i_w, _ = img.shape
        crop_size = self.img_height if self.img_height == self.img_width else None

        # for BSDS 352/BRIND
        if i_w > crop_size and i_h > crop_size:  # later 400, before crop_size
            i = random.randint(0, i_h - crop_size)
            j = random.randint(0, i_w - crop_size)
            img = img[i : i + crop_size, j : j + crop_size]
            gt = gt[i : i + crop_size, j : j + crop_size]

        else:
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.6  # 0.5 for BIPED
        gt = np.clip(gt, 0.0, 1.0)  # BIPED
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt
