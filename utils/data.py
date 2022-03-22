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


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width
    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """

    target = Instances(image_size)


    rotated = all(obj["bbox_mode"] == BoxMode.XYWHA_ABS for obj in annos)
    if rotated:
        boxes = [obj["bbox"] for obj in annos]
        boxes = target.gt_boxes = RotatedBoxes(boxes)
    else:
        boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_1numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks
    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)
    return target
def convert_to_coco_dict(dataset_name):

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        #print(str(image_id) + str(image_dict))
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict["annotations"]
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            # Computing areas using bounding boxes
            bbox_xy = BoxMode.convert(bbox, bbox_mode, BoxMode.XYXY_ABS)
            area = Boxes([bbox_xy]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)
            coco_annotation["category_id"] = reverse_id_mapper(annotation["category_id"])

            # Add optional fields
            coco_annotations.append(coco_annotation)


    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    return coco_dict


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):

    coco_dict = convert_to_coco_dict(dataset_name)

    PathManager.mkdirs(os.path.dirname(output_file))
    with PathManager.open(output_file, "w") as f:
      json.dump(coco_dict, f)



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




