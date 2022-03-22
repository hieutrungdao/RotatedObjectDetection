# import some common libraries
import numpy as np
import copy
import os, json, cv2, random
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
from tqdm import tqdm


import torch

from detectron2.structures import BoxMode

from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils



def rotate_transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
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
        rotate_transform_instance_annotations(obj, transforms, image.shape[:2]) 
        for obj in dataset_dict.pop("annotations")
    ]
    instances = utils.annotations_to_instances_rotated(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


def get_RotatedBox_dict(path):     
    dataset_dicts = []
    for dir in os.listdir(path):
        if os.path.isdir(path+dir):
            for filename in os.listdir(path+dir):
                if filename.endswith(".xml"):
                    portion = os.path.splitext(filename)
                    if not (os.path.exists(path+dir+"/"+portion[0]+".jpg")):
                        # jpg file not found
                        print(path+dir+"/"+filename)
                    else:
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
                                obj["bbox"] = numpy.asarray(obj["bbox"])
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



if __name__ == "__main__":

    from pprint import pprint

    path = "../../data/Phamacity/"
    dataset_dicts = get_RotatedBox_dict(path)
    train_dataset_dicts, test_dataset_dicts, val_dataset_dicts = split_dataset(dataset_dicts)

    print("Dataset size: " + str(len(dataset_dicts)))
    print("Train size: " + str(len(train_dataset_dicts)))
    print("Test size: " + str(len(test_dataset_dicts)))
    print("Val size: " + str(len(val_dataset_dicts)))

    pprint(dataset_dicts[2])




