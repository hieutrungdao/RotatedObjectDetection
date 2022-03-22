# import torch
import copy,torch,torchvision
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from pprint import pprint

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader


from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.structures import RotatedBoxes
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import RotatedCOCOEvaluator,DatasetEvaluators, inference_on_dataset, coco_evaluation,DatasetEvaluator

from utils.config import get_rotated_config
from utils.data import get_RotatedBox_dict

if __name__ == "__main__":

    cfg = get_rotated_config()
    cfg.MODEL.WEIGHTS = "./weights/model_final.pth"

    # predictor = RotatedPredictor(cfg)
    predictor = DefaultPredictor(cfg)

    path = "../../data/Phamacity/"
    dataset_dicts = get_RotatedBox_dict(path)
    for d in random.sample(dataset_dicts, 3):
        pprint(d)
        im = cv2.imread(d["file_name"])

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        
        v = Visualizer(im[:, :, ::-1],
                        metadata=MetadataCatalog.get("Test"), 
                        scale=0.5)
                        # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        # )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("output/" + d["image_id"], out.get_image()[:, :, ::-1])


