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
    confidence_threshold: float = 0.9

    @staticmethod
    def _getValuesFromFile(
        *, json_file_path: Union[str, Path], format=fo.types.COCODetectionDataset
    ):
        coco_dataset_api = coco.COCO(json_file_path)

        img_info = coco_dataset_api.imgs[1]
        img_width = img_info["width"]
        img_height = img_info["height"]
        # Get all image ids
        img_ids = coco_dataset_api.getImgIds()

        # Initialize the list of lists
        bboxes = [
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
                bboxes[index][label].append(
                    [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                )

        return img_height, img_width, bboxes

    @staticmethod
    def _scatter(
        bboxes: List,
        is_ground_truth: bool,
        correctly_predicted: Optional[List] = None,
    ):
        plt.clf()
        if is_ground_truth == False:
            for i in range(len(bboxes)):
                if i % 100 == 0:
                    print(f"Completed {i} out of {len(bboxes)}")
                for j in range(len(bboxes[i])):

                    if correctly_predicted[i][j]:
                        plt.scatter(x=bboxes[i][j][2], y=bboxes[i][j][3], color="g")
                    else:
                        plt.scatter(x=bboxes[i][j][2], y=bboxes[i][j][3], color="r")

            plt.xlabel("Widths of bboxes")
            plt.ylabel("Heights of bboxes")
            plt.grid()
            plt.legend(["Correctly Predicted", "Incorrectly Predicted"])
            plt.title("Scatter plot of bboxes")
            plt.savefig("plots/predictionScatter.png")

        else:

            for i in range(len(bboxes)):
                if i % 100 == 0:
                    print(f"Completed {i} out of {len(bboxes)}")
                for j in range(len(bboxes[i])):
                    plt.scatter(
                        x=bboxes[i][j][2],
                        y=bboxes[i][j][3],
                        color="b",
                    )
            plt.xlabel("Widths of bboxes")
            plt.ylabel("Heights of bboxes")
            plt.grid()
            plt.title("Scatter plot of bboxes")
            plt.savefig("plots/actualScatter.png")

    @staticmethod
    def _histogram(
        bboxes: List,
        is_ground_truth: bool,
    ):
        heights_widths = []
        for i in range(len(bboxes)):
            for j in range(len(bboxes[i])):
                heights_widths.append(
                    (
                        50 * round(bboxes[i][j][3] / 50),
                        50 * round(bboxes[i][j][2] / 50),
                    )
                )
        df = pd.DataFrame(heights_widths)
        categories = df.value_counts().index
        counts = df.value_counts().values
        frequencies = np.array([[categories[i], counts[i]] for i in range(len(counts))])
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
        plt.show()
        if is_ground_truth == True:
            plt.savefig("plots/actualHistogram.png")
        else:
            plt.savefig("plots/predictedHistogram.png")

    @staticmethod
    def _bboxesLoc(
        image_height: np.array,
        image_width: np.array,
        bboxes: List,
        is_ground_truth: bool,
        correctly_predicted: Optional[List] = None,
    ):
        """Plots the location of bboxes"""
        fig, ax = plt.subplots(figsize=(image_width / 96, image_height / 96))
        ax.axes.set_aspect("equal")
        if is_ground_truth == False:
            for i in range(len(bboxes)):
                for j in range(len(bboxes[i])):
                    if correctly_predicted[i][j]:
                        rect = patches.Rectangle(
                            (bboxes[i][j][0] - bboxes[i][j][3], bboxes[i][j][1]),
                            bboxes[i][j][2],
                            bboxes[i][j][3],
                            linewidth=0.3,
                            edgecolor="g",
                            facecolor="none",
                        )
                    else:
                        rect = patches.Rectangle(
                            (bboxes[i][j][0] - bboxes[i][j][3], bboxes[i][j][1]),
                            bboxes[i][j][2],
                            bboxes[i][j][3],
                            linewidth=0.3,
                            edgecolor="r",
                            facecolor="none",
                        )
                    ax.add_patch(rect)

            plt.xlim([0, image_width])
            plt.ylim([0, image_height])
            plt.title("BBoxes Distribution")
            plt.show()
            plt.savefig("plots/predictedBBoxDist.png")
        else:
            for i in range(len(bboxes)):
                for j in range(len(bboxes[i])):
                    rect = patches.Rectangle(
                        (bboxes[i][j][0] - bboxes[i][j][3], bboxes[i][j][1]),
                        bboxes[i][j][2],
                        bboxes[i][j][3],
                        linewidth=0.3,
                        edgecolor="b",
                        facecolor="none",
                    )
                ax.add_patch(rect)

            plt.xlim([0, image_width])
            plt.ylim([0, image_height])
            plt.title("BBoxes Distribution")
            plt.show()
            plt.savefig("plots/actualBBoxDist.png")

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
                    bbox[:4]
                    for bbox in class_bboxes
                    if bbox[4] >= self.confidence_threshold
                ]
                for class_bboxes in image_bboxes
            ]
            for image_bboxes in output_bboxes
        ]

        return filtered_bboxes

    def scatterPlot(
        self,
        *,
        val_file_path: Union[str, Path],
        val_dataloader: Any,
        model: Any,
    ):
        actual_bboxes = BBoxesEDA()
        image_height, image_width, actual_bboxes.bboxes = self._getValuesFromFile(
            json_file_path=val_file_path,
        )

        predicted_bboxes = BBoxesEDA()
        predicted_bboxes.bboxes = self._evaluate(
            model=model,
            val_dataloader=val_dataloader,
        )

        correctly_predicted = predicted_bboxes.compute_iou(
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


class BBoxesEDA(BaseModel):

    bboxes: Optional[List[List[List[float]]]]

    def __init__(self) -> None:
        super().__init__()

    def _categories(self) -> int:
        return len(self.bboxes[0])

    def compute_iou(self, bboxes: BBoxesEDA, threshold: float):

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

                    for l in range(len(bboxes.bboxes)):
                        for m in range(len(bboxes.bboxes[l])):
                            for n in range(len(bboxes.bboxes[l][m])):
                                bbox2 = bboxes.bboxes[l][m][n]
                                iou = self._calculate_iou(
                                    bbox1,
                                    bbox2,
                                )
                                if iou > threshold:
                                    result[i][j][k] = True
                                    break
        return result

    @staticmethod
    def _calculate_iou(
        bbox1: List[float],
        bbox2: List[float],
    ):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersect = (x_right - x_left) * (y_bottom - y_top)
        union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersect

        return intersect / union
