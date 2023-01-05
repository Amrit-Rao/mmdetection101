from segmentation import Visualize
from segmentation import ExperimentManager

dir_path = "data/kitti-coco/validation"
classes = (
    "Car",
    "Cyclist",
    "DontCare",
    "Misc",
    "Pedestrian",
    "Person_sitting",
    "Tram",
    "Truck",
    "Van",
)
type = "CocoDataset"


experiment = ExperimentManager()
experiment.getConfigFromPath(
    config_path="mmdetection/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py",
)
experiment.buildDataset(data_path="data/kitti-coco", classes=classes, type=type)
experiment.buildModel(
    classes=classes,
    checkpoint_path="train_runs/2022-12-20 11:10:03.910932/latest.pth",
)
model = experiment.getModel()
val_dataloader = experiment.getValDataloader()

visualization = Visualize()
visualization.scatterPlot(
    dir_path=dir_path,
    val_dataloader=val_dataloader,
    model=model,
)
