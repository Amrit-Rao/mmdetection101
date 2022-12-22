from segmentation import ExperimentManager
import datetime


current_datetime = datetime.datetime.now()
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

a = ExperimentManager(name=str(current_datetime))
a.getConfigFromPath(
    config_path="mmdetection/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py"
)
a.buildDataset(data_path="data/kitti-coco", classes=classes, type=type)
a.buildModel(
    classes=classes, checkpoint_path="train_runs/2022-12-20 11:10:03.910932/latest.pth"
)
# a.train(output_path="train_runs", num_of_epochs=4)
# a.evaluate(output_path="eval_runs")
a.predict(output_path="predictions")

# out: results.bbox.json and results.segm.json
