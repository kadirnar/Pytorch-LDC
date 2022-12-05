from __future__ import print_function

import argparse
import os

import torch
from torch.utils.data import DataLoader

from model.base_model import LDC
from utils.config_utils import get_config
from utils.dataset import TestDataset
from utils.img_processing import save_image_batch_to_disk


class LdcTest:
    def __init__(self, config_path, single_image=False):
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
        if single_image:
            self.single_image_test()
        else:
            self.dataloader_val()
            self.multi_image_test()

    def load_model(self):
        model = LDC().to(self.device)
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        model.eval()
        print("Number of parameters: {}".format(sum(p.numel() for p in model.parameters())))
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

    def single_image_test(self, image):
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            output = output.cpu().numpy()
        return output

    def multi_image_test(self):
        with torch.no_grad():
            for batch_id, sample_batched in enumerate(self.dataloader_val):
                images = sample_batched["images"].to(self.device)
                file_names = sample_batched["file_names"]
                image_shape = sample_batched["image_shape"]
                preds = self.model(images)
                save_image_batch_to_disk(preds, self.output_dir, file_names, image_shape, is_testing=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="data/config/default_config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()
    ldc_test = LdcTest(args.config)
