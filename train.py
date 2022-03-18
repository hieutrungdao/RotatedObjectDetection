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


def my_transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    if annotation["bbox_mode"] == BoxMode.XYWHA_ABS:
        annotation["bbox"] = transforms.apply_rotated_box(np.asarray([annotation["bbox"]]))[0]
    else:
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

    return annotation

def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with our own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        my_transform_instance_annotations(obj, transforms, image.shape[:2]) 
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances_rotated(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict




class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)
        
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)

class RotatedPredictor(DefaultPredictor):
    def __init__(self, cfg):
        
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = trainer.model
        self.model.eval()

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.transform_gen.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

# As of 0.3 the XYWHA_ABS box is not supported in the visualizer, this is fixed in master branch atm (19/11/20)
class myVisualizer(Visualizer):
  
    def draw_dataset_dict(self, dic):
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYWHA_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(labels=labels, boxes=boxes, masks=masks, keypoints=keypts)

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            sem_seg = cv2.imread(dic["sem_seg_file_name"], cv2.IMREAD_GRAYSCALE)
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output



def get_RotatedBox_dict(path): 
    
    dataset_dicts = []
    
    for dir in os.listdir(path):
        if os.path.isdir(path+dir):
            for filename in os.listdir(path+dir):
                if filename.endswith(".xml"):
                    portion = os.path.splitext(filename)
                    
                    if not (os.path.exists(path+dir+"/"+portion[0]+".jpg")):
                        print(path+dir+"/"+filename)
                        pass

                    record = {}

                    tree = ET.parse(path+dir+"/"+filename, parser=ET.XMLParser(encoding="utf-8"))
                    root = tree.getroot()

                    
                    
                    record["file_name"] = path+dir+"/"+portion[0]+".jpg"
                    record["image_id"] = portion[0]+".jpg"
                    record["width"] = int(root[4][0].text)
                    record["height"] = int(root[4][1].text)
                    record["depth"] = int(root[4][2].text)

                    objs = []
                    for obj in root[6:]:
                        if obj.tag == "object":
                            obj = {
                                "bbox": [float(obj[5][0].text), float(obj[5][1].text), 
                                    float(obj[5][2].text), float(obj[5][3].text), float(obj[5][4].text)*(180/np.pi)],
                                "bbox_mode": BoxMode.XYWHA_ABS,
                                "category_id": 0,
                            }
                        objs.append(obj)
                    record["annotations"] = objs

                    dataset_dicts.append(record)    
    return dataset_dicts


def split_dataset(dataset, train_size=.8, test_size=.1):

    ds_size = int(len(dataset))
    train_size = round(ds_size*train_size)
    test_size = round(ds_size*test_size)

    np.random.shuffle(dataset)

    return dataset[0:train_size], dataset[train_size:(train_size+test_size)], dataset[(train_size+test_size):(ds_size)]



def get_rotated_config():

    cfg = get_cfg()

    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo
    cfg.DATASETS.TRAIN = (["Train"])
    cfg.DATASETS.TEST = (["Test"])

    cfg.MODEL.MASK_ON=False
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
    cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90,-60,-30,0,30,60,90]]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 
    cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   #this is far lower than usual.  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV=4
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV=8
    cfg.SOLVER.IMS_PER_BATCH = 6 #can be up to  24 for a p100 (6 default)
    cfg.SOLVER.CHECKPOINT_PERIOD=10
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.GAMMA=0.5
    cfg.SOLVER.STEPS=[20,40,80]
    cfg.SOLVER.MAX_ITER=100


    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True 
    cfg.DATALOADER.SAMPLER_TRAIN= "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD=0.01
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)#lets just check our output dir exists
    cfg.MODEL.BACKBONE.FREEZE_AT=6


    # Notes on how to implement:
    # https://github.com/facebookresearch/detectron2/issues/21#issuecomment-595522318
    # MODEL:
    #   ANCHOR_GENERATOR:
    #     NAME: RotatedAnchorGenerator
    #     ANGLES: [[-90,-60,-30,0,30,60,90]]
    #   PROPOSAL_GENERATOR:
    #     NAME: RRPN
    #   RPN:
    #     BBOX_REG_WEIGHTS: (1,1,1,1,1)
    #   ROI_BOX_HEAD:
    #     POOLER_TYPE: ROIAlignRotated
    #     BBOX_REG_WEIGHTS: (10,10,5,5,1)
    #   ROI_HEADS:
    #     NAME: RROIHeads

    # print(cfg.OUTPUT_DIR)
    # print(cfg.MODEL)``
    return cfg


def train():
    cfg = get_rotated_config()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


                    
if __name__ == "__main__":

    # im = cv2.imread(("../../data/PhamacityDota/images/linhdam2_10.png"))
    # visualizer = Visualizer(im)
    # out = visualizer.draw_rotated_box_with_label(rotated_box)
    # cv2.imwrite(, out.get_image())

    path = "../../data/Phamacity/"
    dataset_dicts = get_RotatedBox_dict(path)
    train_dataset_dicts, test_dataset_dicts, val_dataset_dicts = split_dataset(dataset_dicts)

    print("Dataset size: " + str(len(dataset_dicts)))
    print("Train size: " + str(len(train_dataset_dicts)))
    print("Test size: " + str(len(test_dataset_dicts)))
    print("Val size: " + str(len(val_dataset_dicts)))

    DatasetCatalog.register("Train", lambda d="Train": train_dataset_dicts)
    MetadataCatalog.get("Train").set(thing_classes=["person"])
    ds_metadata = MetadataCatalog.get("Train")

    DatasetCatalog.register("Test", lambda d="Test": test_dataset_dicts)
    MetadataCatalog.get("Test").set(thing_classes=["person"])

    DatasetCatalog.register("Val", lambda d="Val": val_dataset_dicts)
    MetadataCatalog.get("Val").set(thing_classes=["person"])
    
    # for d in random.sample(train_dataset_dicts, 3):
    #     pprint(d)
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=ds_metadata, scale=0.5)
    #     out = visualizer.draw_dataset_dict(d)
    #     cv2.imwrite("demo/output/" + d["image_id"], out.get_image()[:, :, ::-1])    

    train()


    

  