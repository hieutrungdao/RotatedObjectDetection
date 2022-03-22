
from detectron2.config import get_cfg
from detectron2 import model_zoo


def get_def_config():

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo
    cfg.DATASETS.TRAIN = (["Train"])
    cfg.DATASETS.TEST = ([["Test"]])

    cfg.MODEL.MASK_ON=False
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
    cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90,-60,-30,0,30,60,90]]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 
    cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   #this is far lower than usual.  
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES =len(ClassesNames.keys())
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV=4
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV=8
    cfg.SOLVER.IMS_PER_BATCH = 6#can be up to  24 for a p100 
    cfg.SOLVER.CHECKPOINT_PERIOD=1500
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.GAMMA=0.5
    cfg.SOLVER.STEPS=[1000,2000,4000,8000, 12000]
    cfg.SOLVER.MAX_ITER=14000


    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True 
    cfg.DATALOADER.SAMPLER_TRAIN= "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD=0.01
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)#lets just check our output dir exists
    cfg.MODEL.BACKBONE.FREEZE_AT=6
    print(cfg.MODEL)

    return cfg


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
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1,1,1,1,1)
    
    cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90,-60,-30,0,30,60,90]]

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   #this is far lower than usual.  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV=4
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV=8

    cfg.SOLVER.IMS_PER_BATCH = 6 #can be up to  24 for a p100 (6 default)
    cfg.SOLVER.CHECKPOINT_PERIOD=10
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.GAMMA=0.5
    cfg.SOLVER.STEPS=[5,20,40,80]
    cfg.SOLVER.MAX_ITER=10

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
