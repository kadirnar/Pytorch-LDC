import argparse
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader

from model.base_model import LDC
from utils.config_utils import get_config
from utils.data_utils import create_dir, torch_to_numpy
from utils.dataset import TestDataset


class LdcEval:
    def __init__(self, config_path):
        config = get_config(config_path)
        self.output_dir = config.DATASET.OUTPUT_DIR
        self.train_data = config.DATASET.TRAIN_DATA
        checkpoint_data = config.TRAIN_CONFIG.CHECKPOINT_DATA
        self.checkpoint_path = os.path.join(checkpoint_data)
        self.input_dir = config.DATASET.INPUT_DIR
        self.test_data = config.DATASET.TEST_DATA
        self.mean_pixel_values = config.TRAIN_CONFIG.MEAN_PIXEL_VALUES
        self.test_list = config.DATASET.TEST_LIST
        self.num_worker = config.TRAIN_CONFIG.NUM_WORKER
        self.device = config.TRAIN_CONFIG.DEVICE
        self.img_width = config.DATASET.IMG_WIDTH
        self.img_height = config.DATASET.IMG_HEIGHT

        self.load_model()
        self.multi_image_test()

    def load_model(self):
        model = LDC().to(self.device)
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        model.eval()
        self.model = model

    def dataloader_val(self):
        dataset_val = TestDataset(
            self.input_dir,
            img_width=self.img_width,
            img_height=self.img_height,
            mean_bgr=self.mean_pixel_values[0:3] if len(self.mean_pixel_values) == 4 else self.mean_pixel_values,
            test_data=self.test_data,
            test_list=self.test_list,
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_worker,
        )
        self.dataloader_val = dataloader_val

    def one_channel_image_label_save(self, image, label, image_name, label_name):
        create_dir(self.output_dir + "/result")
        create_dir(self.output_dir + "/result/image")
        create_dir(self.output_dir + "/result/label")
        cv2.imwrite(self.output_dir + "/result/image/" + image_name, image)
        cv2.imwrite(self.output_dir + "/result/label/" + label_name, label)

    def multi_image_test(self):
        self.dataloader_val()
        precision_list = []
        recall_list = []
        with torch.no_grad():
            for i, sample_batched in enumerate(self.dataloader_val):
                images = sample_batched["images"].to(self.device)
                labels = sample_batched["labels"].to(self.device)
                outputs = self.model(images)
                for _, output in enumerate(outputs):
                    output = torch_to_numpy(output)
                    label = torch_to_numpy(labels)
                    self.one_channel_image_label_save(output, label, f"image_{i}.png", f"label_{i}.png")
                    precision = precision_score(label.flatten(), output.flatten())
                    recall = recall_score(label.flatten(), output.flatten())
                    precision_list.append(precision)
                    recall_list.append(recall)
        print(f"Precision: {np.mean(precision_list)}")
        print(f"Recall: {np.mean(recall_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="data/config/default_config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()
    ldc_test = LdcEval(args.config)
    print(ldc_test)
