
import os
import cv2
import random
import copy
import numpy as np
import multiprocessing

import torch
import torchvision

import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

from utils.data import get_RotatedBox_dict, split_dataset
from utils.config import get_rotated_config
from utils.engine import RotatedTrainer


def train():
    cfg = get_rotated_config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = RotatedTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()


                    
if __name__ == "__main__":
    
    from pprint import pprint

    multiprocessing.set_start_method("spawn", force=True)

    setup_logger(name="fvcore")
    logger = setup_logger()

    logger.info("Data preparation")
    path = "../../data/Phamacity/"
    dataset_dicts = get_RotatedBox_dict(path)
    train_dataset_dicts, test_dataset_dicts, val_dataset_dicts = split_dataset(dataset_dicts, test_size=.01)
    
    logger.info("Dataset size: " + str(len(dataset_dicts)))
    logger.info("Train size: " + str(len(train_dataset_dicts)))
    logger.info("Test size: " + str(len(test_dataset_dicts)))
    logger.info("Val size: " + str(len(val_dataset_dicts)))

    DatasetCatalog.register("Train", lambda d="Train": train_dataset_dicts)
    MetadataCatalog.get("Train").set(thing_classes=["person"])
    ds_metadata = MetadataCatalog.get("Train")

    DatasetCatalog.register("Test", lambda d="Test": test_dataset_dicts)
    MetadataCatalog.get("Test").set(thing_classes=["person"])

    DatasetCatalog.register("Val", lambda d="Val": val_dataset_dicts)
    MetadataCatalog.get("Val").set(thing_classes=["person"])
    
    pprint(train_dataset_dicts[1])

    train()


    

  