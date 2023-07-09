from __future__ import annotations
import json
import matplotlib.pyplot as plt
from pydantic import BaseModel
from typing import Union, Any, Optional, List, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from mmdet.apis import single_gpu_test
from mmdet.utils import build_dp
import fiftyone as fo
import pycocotools.coco as coco
from tqdm import tqdm

primitive_type = Union[int, float, str, bool]


class Visualize(BaseModel):

    iou_threshold: float = 0.7
    confidence_threshold: float = 0.8

    @staticmethod
    def _getValuesFromFile(
        *, json_file_path: Union[str, Path], format=fo.types.COCODetectionDataset
    ):
        coco_dataset_api = coco.COCO(json_file_path)
        categories = coco_dataset_api.loadCats(coco_dataset_api.getCatIds())
        categories_dict = {category["id"]: category["name"] for category in categories}
        img_info = coco_dataset_api.imgs[1]
        img_width = img_info["width"]
        img_height = img_info["height"]
        # Get all image ids
        img_ids = coco_dataset_api.getImgIds()

        # Initialize the list of lists
        bboxes_list = [
            [[] for _ in range(len(coco_dataset_api.cats))] for _ in range(len(img_ids))
        ]

        # Iterate over the images
        for index, img_id in enumerate(img_ids):
            # Get the annotations for the current image
            ann_ids = coco_dataset_api.getAnnIds(imgIds=img_id)
            annotations = coco_dataset_api.loadAnns(ann_ids)

            # Iterate over the annotations
            for annotation in annotations:
                label = annotation["category_id"]
                box = annotation["bbox"]
                bboxes_list[index][label].append(
                    BBox(
                        min_x=box[0],
                        min_y=box[1],
                        max_x=box[0] + box[2],
                        max_y=box[1] + box[3],
                        image_id=img_id,
                        category_id=label,
                        is_ground_truth=True,
                    )
                )
        return img_height, img_width, categories_dict, bboxes_list

    def _evaluate(
        self,
        *,
        model: Any,
        val_dataloader: Any,
    ):
        model_dp = build_dp(model, "cuda", device_ids=range(1))
        output_bboxes = single_gpu_test(
            model_dp, val_dataloader, show=False, show_score_thr=0.3
        )
        filtered_bboxes = [
            [
                [
                    BBox(
                        min_x=bbox[0],
                        min_y=bbox[1],
                        max_x=bbox[2],
                        max_y=bbox[3],
                        image_id=image_id,
                        category_id=category_id,
                        is_ground_truth=False,
                    )
                    for bbox in class_bboxes
                    if bbox[4] >= self.confidence_threshold
                ]
                for category_id, class_bboxes in enumerate(image_bboxes)
            ]
            for image_id, image_bboxes in enumerate(output_bboxes)
        ]

        return filtered_bboxes

    @staticmethod
    def _scatter(
        bboxes: BBoxesEDA,
        is_ground_truth: bool,
        correctly_predicted: Optional[List] = None,
    ):
        for category_id in bboxes.categories_dict.keys():
            plt.clf()
            if is_ground_truth == False:
                for i in tqdm(range(len(bboxes.bboxes))):
                    for j in range(len(bboxes.bboxes[i])):
                        if j == category_id:
                            for k in range(len(bboxes.bboxes[i][j])):
                                if correctly_predicted[i][j][k]:
                                    plt.scatter(
                                        x=bboxes.bboxes[i][j][k].width,
                                        y=bboxes.bboxes[i][j][k].height,
                                        color="g",
                                    )
                                else:
                                    plt.scatter(
                                        x=bboxes.bboxes[i][j][k].width,
                                        y=bboxes.bboxes[i][j][k].height,
                                        color="r",
                                    )

                plt.xlabel("Widths of bboxes")
                plt.ylabel("Heights of bboxes")
                plt.grid()
                plt.legend(["Correctly Predicted", "Incorrectly Predicted"])
                plt.title("Scatter plot of bboxes")
                plt.savefig(
                    f"plots/predictionScatter_{bboxes.categories_dict[category_id]}.png"
                )

            else:

                for i in tqdm(range(len(bboxes.bboxes))):
                    for j in range(len(bboxes.bboxes[i])):
                        if j == category_id:
                            for k in range(len(bboxes.bboxes[i][j])):
                                plt.scatter(
                                    x=bboxes.bboxes[i][j][k].width,
                                    y=bboxes.bboxes[i][j][k].height,
                                    color="b",
                                )
                plt.xlabel("Widths of bboxes")
                plt.ylabel("Heights of bboxes")
                plt.grid()
                plt.title("Scatter plot of bboxes")
                plt.savefig(
                    f"plots/actualScatter_{bboxes.categories_dict[category_id]}.png"
                )

    @staticmethod
    def _histogram(
        bboxes: BBoxesEDA,
        is_ground_truth: bool,
    ):
        for category_id in bboxes.categories_dict.keys():
            plt.clf()
            heights_widths = []
            areas = []
            for i in tqdm(range(len(bboxes.bboxes))):
                for j in range(len(bboxes.bboxes[i])):
                    if j == category_id:
                        for k in range(len(bboxes.bboxes[i][j])):
                            heights_widths.append(
                                (
                                    50 * round(bboxes.bboxes[i][j][k].height / 50),
                                    50 * round(bboxes.bboxes[i][j][k].width / 50),
                                )
                            )
                            areas.append(bboxes.bboxes[i][j][k].get_area())
            df = pd.DataFrame(heights_widths)
            categories = df.value_counts().index
            counts = df.value_counts().values
            frequencies = np.array(
                [[categories[i], counts[i]] for i in range(len(counts))]
            )
            frequencies = frequencies[frequencies[:, 0].argsort()]
            fig, ax = plt.subplots()
            ranges = []
            for x in frequencies[:, 0]:
                if x[0] == 0 and x[1] == 0:
                    ranges.append(f"(({x[0]},{x[0]+25}),({x[1]},{x[1]+25}))")
                elif x[0] == 0:
                    ranges.append(f"(({x[0]},{x[0]+25}),({x[1]-25},{x[1]+25}))")
                elif x[1] == 0:
                    ranges.append(f"(({x[0]-25},{x[0]+25}),({x[1]},{x[1]+25}))")
                else:
                    ranges.append(f"(({x[0]-25},{x[0]+25}),({x[1]-25},{x[1]+25}))")
            ax.bar(
                ranges,
                np.int64(frequencies[:, 1]),
            )
            plt.setp(ax.get_xticklabels(), rotation=90)
            ax.set_title("Histogram of sizes of bboxes")
            plt.xticks(fontsize=8)
            if is_ground_truth == True:
                plt.savefig(
                    f"plots/actualHistogramHeightWidth_{bboxes.categories_dict[category_id]}.png"
                )
            else:
                plt.savefig(
                    f"plots/predictedHistogramHeightWidth_{bboxes.categories_dict[category_id]}.png"
                )

            plt.clf()
            plt.hist(areas)
            plt.xlabel("Area")
            plt.ylabel("Count")
            plt.grid()
            plt.title("Histogram of areas")
            if is_ground_truth == True:
                plt.savefig(
                    f"plots/actualHistogramAreas_{bboxes.categories_dict[category_id]}.png"
                )
            else:
                plt.savefig(
                    f"plots/predictionHistogramAreas_{bboxes.categories_dict[category_id]}.png"
                )

    @staticmethod
    def _bboxesLoc(
        image_height: float,
        image_width: float,
        bboxes: BBoxesEDA,
        is_ground_truth: bool,
        correctly_predicted: Optional[List] = None,
    ):
        """Plots the location of bboxes"""
        for category_id in bboxes.categories_dict.keys():
            plt.clf()
            fig, ax = plt.subplots(figsize=(image_width / 96, image_height / 96))
            ax.axes.set_aspect("equal")
            if is_ground_truth == False:
                for i in tqdm(range(len(bboxes.bboxes))):
                    for j in range(len(bboxes.bboxes[i])):
                        if j == category_id:
                            for k in range(len(bboxes.bboxes[i][j])):
                                if correctly_predicted[i][j][k]:
                                    rect = patches.Rectangle(
                                        (
                                            bboxes.bboxes[i][j][k].max_x,
                                            image_height - bboxes.bboxes[i][j][k].max_y,
                                        ),
                                        bboxes.bboxes[i][j][k].width,
                                        bboxes.bboxes[i][j][k].height,
                                        linewidth=0.3,
                                        edgecolor="g",
                                        facecolor="none",
                                    )
                                else:
                                    rect = patches.Rectangle(
                                        (
                                            bboxes.bboxes[i][j][k].max_x,
                                            image_height - bboxes.bboxes[i][j][k].max_y,
                                        ),
                                        bboxes.bboxes[i][j][k].width,
                                        bboxes.bboxes[i][j][k].height,
                                        linewidth=0.3,
                                        edgecolor="r",
                                        facecolor="none",
                                    )
                                ax.add_patch(rect)

                plt.xlim([0, image_width])
                plt.ylim([0, image_height])
                plt.title("BBoxes Distribution")
                plt.savefig(
                    f"plots/predictedBBoxDist_{bboxes.categories_dict[category_id]}.png"
                )
            else:
                for i in tqdm(range(len(bboxes.bboxes))):
                    for j in range(len(bboxes.bboxes[i])):
                        if j == category_id:
                            for k in range(len(bboxes.bboxes[i][j])):
                                rect = patches.Rectangle(
                                    (
                                        bboxes.bboxes[i][j][k].max_x,
                                        image_height - bboxes.bboxes[i][j][k].max_y,
                                    ),
                                    bboxes.bboxes[i][j][k].width,
                                    bboxes.bboxes[i][j][k].height,
                                    linewidth=0.3,
                                    edgecolor="b",
                                    facecolor="none",
                                )
                                ax.add_patch(rect)

                plt.xlim([0, image_width])
                plt.ylim([0, image_height])
                plt.title("BBoxes Distribution")
                plt.savefig(
                    f"plots/actualBBoxDist_{bboxes.categories_dict[category_id]}.png"
                )

    def scatterPlot(
        self,
        *,
        val_file_path: Union[str, Path],
        val_dataloader: Any,
        model: Any,
    ):
        actual_bboxes = BBoxesEDA()
        (
            image_height,
            image_width,
            actual_bboxes.categories_dict,
            actual_bboxes.bboxes,
        ) = self._getValuesFromFile(
            json_file_path=val_file_path,
        )
        predicted_bboxes = BBoxesEDA()
        predicted_bboxes.bboxes = self._evaluate(
            model=model,
            val_dataloader=val_dataloader,
        )
        predicted_bboxes.categories_dict = actual_bboxes.categories_dict
        correctly_predicted = predicted_bboxes.compute_iou_of_bboxes(
            actual_bboxes,
            self.iou_threshold,
        )

        self._scatter(actual_bboxes, is_ground_truth=True)
        self._scatter(
            predicted_bboxes,
            is_ground_truth=False,
            correctly_predicted=correctly_predicted,
        )

        return self

    def bboxLoc(
        self,
        *,
        val_file_path: Union[str, Path],
        val_dataloader: Any,
        model: Any,
    ):
        actual_bboxes = BBoxesEDA()
        (
            image_height,
            image_width,
            actual_bboxes.categories_dict,
            actual_bboxes.bboxes,
        ) = self._getValuesFromFile(
            json_file_path=val_file_path,
        )

        predicted_bboxes = BBoxesEDA()
        predicted_bboxes.bboxes = self._evaluate(
            model=model,
            val_dataloader=val_dataloader,
        )
        predicted_bboxes.categories_dict = actual_bboxes.categories_dict

        correctly_predicted = predicted_bboxes.compute_iou_of_bboxes(
            actual_bboxes,
            self.iou_threshold,
        )

        self._bboxesLoc(
            image_height=image_height,
            image_width=image_width,
            bboxes=actual_bboxes,
            is_ground_truth=True,
        )

        self._bboxesLoc(
            image_height=image_height,
            image_width=image_width,
            bboxes=predicted_bboxes,
            is_ground_truth=False,
            correctly_predicted=correctly_predicted,
        )

        return self

    def histogram(
        self,
        *,
        val_file_path: Union[str, Path],
        val_dataloader: Any,
        model: Any,
    ):
        actual_bboxes = BBoxesEDA()
        (
            image_height,
            image_width,
            actual_bboxes.categories_dict,
            actual_bboxes.bboxes,
        ) = self._getValuesFromFile(
            json_file_path=val_file_path,
        )

        predicted_bboxes = BBoxesEDA()
        predicted_bboxes.bboxes = self._evaluate(
            model=model,
            val_dataloader=val_dataloader,
        )
        predicted_bboxes.categories_dict = actual_bboxes.categories_dict

        self._histogram(actual_bboxes, is_ground_truth=True)

        self._histogram(
            predicted_bboxes,
            is_ground_truth=False,
        )

        return self


class BBox(BaseModel):

    max_x: Optional[float]
    min_x: Optional[float]
    max_y: Optional[float]
    min_y: Optional[float]
    width: Optional[float]
    height: Optional[float]
    image_id: Optional[int]
    category_id: Optional[int]
    ground_truth: Optional[bool]

    def __init__(
        self,
        max_x: float,
        max_y: float,
        min_x: float,
        min_y: float,
        image_id: int,
        category_id: int,
        is_ground_truth: bool,
    ):
        super().__init__()
        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y
        self.image_id = image_id
        self.category_id = category_id
        self.width = max_x - min_x
        self.height = max_y - min_y
        self.ground_truth = is_ground_truth

    def get_area(self) -> float:
        return self.width * self.height

    def get_coordinates(self) -> List[float]:
        return [self.min_x, self.min_y, self.max_x, self.max_y]

    def calculate_iou(
        self,
        bbox: BBox,
    ):
        x_left = max(self.min_x, bbox.min_x)
        y_bottom = max(self.min_y, bbox.min_y)
        x_right = min(self.max_x, bbox.max_x)
        y_top = min(self.max_y, bbox.max_y)

        if x_right < x_left or y_top < y_bottom:
            return 0.0

        intersect = (x_right - x_left) * (y_top - y_bottom)
        union = self.get_area() + bbox.get_area() - intersect

        return intersect / union


class BBoxesEDA(BaseModel):

    bboxes: Optional[List[List[BBox]]]
    categories_dict: Optional[Dict]

    def __init__(self) -> None:
        super().__init__()

    def compute_iou_of_bboxes(self, bboxes: BBoxesEDA, threshold: float):

        result = [
            [
                [False for _ in range(len(img_class_bboxes))]
                for img_class_bboxes in img_classes
            ]
            for img_classes in self.bboxes
        ]

        for i in tqdm(range(len(self.bboxes))):
            for j in range(len(self.bboxes[i])):
                for k in range(len(self.bboxes[i][j])):
                    bbox1 = self.bboxes[i][j][k]
                    flag = False
                    for l in range(len(bboxes.bboxes)):
                        for m in range(len(bboxes.bboxes[l])):
                            for n in range(len(bboxes.bboxes[l][m])):
                                if i == l and j == m:
                                    bbox2 = bboxes.bboxes[l][m][n]
                                    iou = bbox1.calculate_iou(bbox2)
                                    if iou > threshold:
                                        result[i][j][k] = True
                                        flag = True
                                    if flag:
                                        break
                            if flag:
                                break
                        if flag:
                            break

        return result
