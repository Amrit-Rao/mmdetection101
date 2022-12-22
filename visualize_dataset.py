from segmentation import Visualize

json_file_path = "data/kitti-coco/validation/labels.json"

a = Visualize()
a.getJsonFile(file_path=json_file_path)
a.bboxesDist()
