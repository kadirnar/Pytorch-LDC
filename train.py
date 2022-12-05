import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.base_model import LDC
from model.loss import cats_loss
from utils.config_utils import get_config
from utils.dataset import BipedDataset
from utils.img_processing import count_parameters


def main():
    config = get_config("data/config/default_config.yaml")
    device = config.TRAIN_CONFIG.DEVICE
    mean_pixel_values = config.TRAIN_CONFIG.MEAN_PIXEL_VALUES

    # Instantiate model and move it to the computing device
    model = LDC().to(device)
    print("Number of parameters: {}".format(count_parameters(model)))
    ini_epoch = 0
    dataset_train = BipedDataset(
        config.DATASET.INPUT_DIR,
        img_width=config.DATASET.IMG_WIDTH,
        img_height=config.DATASET.IMG_HEIGHT,
        mean_bgr=mean_pixel_values[0:3] if len(mean_pixel_values) == 4 else mean_pixel_values,
        train_mode="train",
        train_data=config.DATASET.TRAIN_DATA,
        train_list=config.DATASET.TRAIN_LIST,
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.TRAIN_CONFIG.BATCH_SIZE,
        shuffle=True,
        num_workers=config.TRAIN_CONFIG.NUM_WORKER,
    )

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.0)

    k = 0
    set_lr = [25e-4, 5e-4, 1e-5]  # [25e-4, 5e-6]
    for epoch in range(ini_epoch, config.TRAIN_CONFIG.EPOCHS):
        if config.TRAIN_CONFIG.ADJUST_LR is not None:
            if epoch in config.TRAIN_CONFIG.ADJUST_LR:
                lr2 = set_lr[k]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr2
                k += 1

        # Create output directories
        output_dir_epoch = os.path.join(config.DATASET.OUTPUT_DIR, config.DATASET.TRAIN_DATA, str(epoch))
        os.makedirs(output_dir_epoch, exist_ok=True)
        print(f"Counting parameters of the model: {count_parameters(model)}")
        model.train()

        l_weight = [
            [0.05, 2.0],
            [0.05, 2.0],
            [0.05, 2.0],
            [0.1, 1.0],
            [0.1, 1.0],
            [0.1, 1.0],
            [0.01, 4.0],
        ]  # for cats loss
        loss_avg = []
        for batch_id, sample_batched in enumerate(dataloader_train):
            images = sample_batched["images"].to(device)  # BxCxHxW
            labels = sample_batched["labels"].to(device)  # BxHxW
            preds_list = model(images)
            # focal loss
            loss = sum([cats_loss(preds, labels, l_w, device) for preds, l_w in zip(preds_list, l_weight)])  # cats_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg.append(loss.item())

            print(
                "Epoch: {} | Batch/Total: {}/{} | Loss: {:.4f} | Avg Loss: {:.4f}".format(
                    epoch, batch_id, len(dataloader_train), loss.item(), np.mean(loss_avg)
                )
            )

        loss_avg = np.array(loss_avg).mean()

        torch.save(
            model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            os.path.join(output_dir_epoch, "{0}_model.pth".format(epoch)),
        )


if __name__ == "__main__":
    main()
