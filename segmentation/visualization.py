from __future__ import annotations
import json
import matplotlib.pyplot as plt
from pydantic import BaseModel
from typing import Union, Any, Optional, List, Dict
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from mmdet.apis import single_gpu_test
from mmdet.utils import build_dp
import fiftyone as fo
import fiftyone.core.metadata as md
import pycocotools.mask as mask

primitive_type = Union[int, float, str, bool]


class Visualize(BaseModel):

    iou_threshold: float = 0.5
    confidence_threshold: float = 0.7

    @staticmethod
    def _getValuesFromFile(
        *,
        dir_path: Union[str, Path],
        format="coco",
    ):

        dataset = fo.load_dataset(dir_path, fmt=format)
        max_labels = md.get_label_set_size(dataset)

        bboxes = [[[] for _ in range(max_labels)] for _ in range(len(dataset))]

        for i, sample in enumerate(dataset):
            boxes = md.get_boxes(sample)
            labels = md.get_labels(sample)
            for box, label in zip(boxes, labels):
                coordinates = [box.x1, box.y1, box.x2, box.y2]
                bbox_obj = BBox(coordinates)
                bboxes[i][label].append(bbox_obj)

        image_width, image_height = fo.utils.metadata.get_image_size(dataset[0])

        return image_width, image_height, bboxes

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
                [bbox for bbox in class_bboxes if bbox[4] >= self.confidence_threshold]
                for class_bboxes in image_bboxes
            ]
            for image_bboxes in output_bboxes
        ]

        return filtered_bboxes

    def scatterPlot(
        self,
        *,
        dir_path: Union[str, Path],
        val_dataloader: Any,
        model: Any,
    ):
        actual_bboxes = BBoxesEDA()
        image_height, image_width, actual_bboxes.bboxes = self._getValuesFromFile(
            dir_path=dir_path
        )
        predicted_bboxes = BBoxesEDA()
        predicted_bboxes.bboxes = self._evaluate(model, val_dataloader)

        correctly_predicted = predicted_bboxes.compute_iou(actual_bboxes)

        self._scatter(actual_bboxes, is_ground_truth=True)
        self._scatter(
            predicted_bboxes,
            is_ground_truth=False,
            correctly_predicted=correctly_predicted,
        )
        return self


class BBox(BaseModel):
    coordinates: Optional[List[float]]

    def __init__(self, coordinates):
        super().__init__()
        self.coordinates = coordinates

    def _IoUvalue(self, *, bbox: BBox):

        iou = mask(bbox.coordinates, self.coordinates, True)

        return iou


class BBoxesEDA(BaseModel):

    bboxes: Optional[List[List[BBox]]]

    def __init__(self, bboxes) -> None:
        super().__init__()
        self.bboxes = bboxes

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

        # Iterate through each image of lst1
        for i, j, k in product(
            range(len(self.bboxes)),
            range(len(self.bboxes[i])),
            range(len(self.bboxes[i][j])),
        ):
            bbox1 = self.bboxes[i][j][k]
            for l, m, n in product(
                range(len(bboxes.bboxes)),
                range(len(bboxes.bboxes[l])),
                range(len(bboxes.bboxes[l][m])),
            ):
                bbox2 = bboxes.bboxes[l][m][n]
                iou = bbox1._IoUvalue(bbox2)
                if iou > self.bbox_threshold:
                    result[i][j][k] = True
                    break
        return result
