from __future__ import annotations
import json
import matplotlib.pyplot as plt
from pydantic import BaseModel
from typing import Union, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from mmdet.apis import single_gpu_test
from mmdet.utils import build_dp

primitive_type = Union[int, float, str, bool]


class Visualize(BaseModel):
    @staticmethod
    def _IoU(
        actual_bbox_x: np.array,
        actual_bbox_y: np.array,
        actual_bbox_heights: np.array,
        actual_bbox_widths: np.array,
        predicted_bbox_x: np.array,
        predicted_bbox_y: np.array,
        predicted_bbox_heights: np.array,
        predicted_bbox_widths: np.array,
    ):
        max_bbox_x = np.maximum(actual_bbox_x, predicted_bbox_x)
        min_bbox_y = np.minimum(actual_bbox_y, predicted_bbox_y)
        min_bbox_x = np.maximum(
            actual_bbox_x + actual_bbox_widths, predicted_bbox_x + predicted_bbox_widths
        )
        max_bbox_y = np.maximum(
            actual_bbox_x - actual_bbox_heights, predicted_bbox_x - actual_bbox_widths
        )
        intersection = np.multiply(max_bbox_x - min_bbox_x, max_bbox_y - min_bbox_y)
        union = (
            np.multiply(actual_bbox_widths, actual_bbox_heights)
            + np.multiply(predicted_bbox_widths, predicted_bbox_heights)
            - intersection
        )

        IoU = np.divide(intersection, union)
        correctly_predicted = IoU > 0.5
        return correctly_predicted

    @staticmethod
    def _getValuesFromFile(file_path: Union[str, Path]):
        with open(file_path, "r") as f:
            config = json.load(f)
        bbox_x = np.zeros((len(config["annotations"]),))
        bbox_y = np.zeros((len(config["annotations"]),))
        bbox_heights = np.zeros((len(config["annotations"]),))
        bbox_widths = np.zeros((len(config["annotations"]),))
        for i in range(len(config["annotations"])):
            bbox_x[i] = config["annotations"][i]["bbox"][0]
            bbox_y[i] = config["annotations"][i]["bbox"][1]
            bbox_widths[i] = config["annotations"][i]["bbox"][2]
            bbox_heights[i] = config["annotations"][i]["bbox"][3]
        image_height = config["images"][1]["height"]
        image_width = config["images"][1]["width"]

        return image_height, image_width, bbox_x, bbox_y, bbox_heights, bbox_widths

    @staticmethod
    def _scatter(
        bbox_widths: np.array,
        bbox_heights: np.array,
        is_ground_truth: bool,
        correctly_predicted: Optional[np.array],
    ):
        if is_ground_truth == False:
            correctly_predicted_widths = bbox_widths[correctly_predicted == True]
            correctly_predicted_heights = bbox_heights[correctly_predicted == True]
            incorrectly_predicted_widths = bbox_widths[correctly_predicted == False]
            incorrectly_predicted_heights = bbox_heights[correctly_predicted == False]
            plt.scatter(
                x=correctly_predicted_widths, y=correctly_predicted_heights, color="g"
            )
            plt.scatter(
                x=incorrectly_predicted_widths,
                y=incorrectly_predicted_heights,
                color="r",
            )
            plt.xlabel("Widths of bboxes")
            plt.ylabel("Heights of bboxes")
            plt.grid()
            plt.legend(["Correctly Predicted", "Incorrectly Predicted"])
            plt.title("Scatter plot of bboxes")
            plt.show()
            plt.savefig("plots/predictionScatter.png")
        else:
            plt.scatter(
                x=bbox_widths,
                y=bbox_widths,
                color="b",
            )
            plt.xlabel("Widths of bboxes")
            plt.ylabel("Heights of bboxes")
            plt.grid()
            plt.title("Scatter plot of bboxes")
            plt.show()
            plt.savefig("plots/actualScatter.png")

    @staticmethod
    def _histogram(
        bbox_heights: np.array, bbox_widths: np.array, is_ground_truth: bool
    ):
        heights_widths = []
        for i in range(len(bbox_heights)):
            heights_widths.append(
                (
                    50 * round(bbox_heights[i] / 50),
                    50 * round(bbox_widths[i] / 50),
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
    def _bboxesDist(
        image_height: np.array,
        image_width: np.array,
        bbox_heights: np.array,
        bbox_widths: np.array,
        bbox_x: np.array,
        bbox_y: np.array,
        is_ground_truth: bool,
        correctly_predicted: Optional[np.array],
    ):
        fig, ax = plt.subplots(figsize=(image_width / 96, image_height / 96))
        ax.axes.set_aspect("equal")
        if is_ground_truth == False:
            for i in range(len(bbox_heights)):
                if correctly_predicted[i] == True:
                    rect = patches.Rectangle(
                        (bbox_x[i] - bbox_heights[i], bbox_y[i]),
                        bbox_widths[i],
                        bbox_heights[i],
                        linewidth=0.3,
                        edgecolor="g",
                        facecolor="none",
                    )
                else:
                    rect = patches.Rectangle(
                        (bbox_x[i] - bbox_heights[i], bbox_y[i]),
                        bbox_widths[i],
                        bbox_heights[i],
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
            for i in range(len(bbox_heights)):
                rect = patches.Rectangle(
                    (bbox_x[i] - bbox_heights[i], bbox_y[i]),
                    bbox_widths[i],
                    bbox_heights[i],
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

    def scatterPlot(
        self, *, val_json_path: Union[str, Path], val_dataloader: Any, model: Any
    ):

        (
            image_height,
            image_width,
            actual_bbox_x,
            actual_bbox_y,
            actual_bbox_heights,
            actual_bbox_widths,
        ) = self._getValuesFromFile(file_path=val_json_path)

        model_dp = build_dp(model, "cuda", device_ids=range(1))
        results = single_gpu_test(
            model_dp, val_dataloader, show=False, show_score_thr=0.3
        )
        print(actual_bbox_widths.shape)

        # predicted_bbox_x = results[:, 0]
        # predicted_bbox_y = results[:, 1]
        # predicted_bbox_widths = results[:, 2]
        # predicted_bbox_heights = results[:, 3]

        # correctly_predicted = self._IoU(
        #     actual_bbox_x,
        #     actual_bbox_y,
        #     actual_bbox_heights,
        #     actual_bbox_widths,
        #     predicted_bbox_x,
        #     predicted_bbox_y,
        #     predicted_bbox_heights,
        #     predicted_bbox_widths,
        # )

        # self._scatter(actual_bbox_widths, actual_bbox_heights, is_ground_truth=True)
        # self._scatter(
        #     predicted_bbox_widths,
        #     predicted_bbox_heights,
        #     is_ground_truth=False,
        #     correctly_predicted=correctly_predicted,
        # )

        return self
