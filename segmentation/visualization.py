from __future__ import annotations
import json
import matplotlib.pyplot as plt
from pydantic import BaseModel
from typing import Union, Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.patches as patches

primitive_type = Union[int, float, str, bool]


class Visualize(BaseModel):
    data_config: Optional[
        Dict[
            str, Union[primitive_type, List[primitive_type], Dict[str, primitive_type]]
        ]
    ]
    image_height: Optional[int]
    image_width: Optional[int]
    bbox_x: Optional[Any]
    bbox_y: Optional[Any]
    bbox_heights: Optional[Any]
    bbox_widths: Optional[Any]

    def getJsonFile(self, *, file_path: Union[str, Path]):
        with open(file_path, "r") as f:
            self.data_config = json.load(f)
        return self

    def _getImageHeightWidth(self):
        self.image_height = self.data_config["images"][1]["height"]
        self.image_width = self.data_config["images"][1]["width"]
        return self

    def _getBBoxes(self):
        self.bbox_x = np.zeros((len(self.data_config["annotations"]),))
        self.bbox_y = np.zeros((len(self.data_config["annotations"]),))
        self.bbox_heights = np.zeros((len(self.data_config["annotations"]),))
        self.bbox_widths = np.zeros((len(self.data_config["annotations"]),))
        for i in range(len(self.data_config["annotations"])):
            self.bbox_x[i] = self.data_config["annotations"][i]["bbox"][0]
            self.bbox_y[i] = self.data_config["annotations"][i]["bbox"][1]
            self.bbox_widths[i] = self.data_config["annotations"][i]["bbox"][2]
            self.bbox_heights[i] = self.data_config["annotations"][i]["bbox"][3]

    def scatterPlot(self):
        self._getImageHeightWidth()
        self._getBBoxes()
        plt.scatter(x=self.bbox_widths, y=self.bbox_heights)
        plt.xlabel("Widths of bboxes")
        plt.ylabel("Heights of bboxes")
        plt.grid()
        plt.title("Scatter plot of bboxes")
        plt.show()
        plt.savefig("plots/scatter.png")

    def histogram(self):
        self._getImageHeightWidth()
        self._getBBoxes()
        heights_widths = []
        for i in range(len(self.bbox_heights)):
            heights_widths.append(
                (
                    50 * round(self.bbox_heights[i] / 50),
                    50 * round(self.bbox_widths[i] / 50),
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
        plt.savefig("plots/histogram.png")

    def bboxesDist(self):
        self._getImageHeightWidth()
        self._getBBoxes()
        fig, ax = plt.subplots(figsize=(self.image_width / 96, self.image_height / 96))
        ax.axes.set_aspect("equal")

        for i in range(len(self.bbox_heights)):
            rect = patches.Rectangle(
                (self.bbox_x[i] - self.bbox_heights[i], self.bbox_y[i]),
                self.bbox_widths[i],
                self.bbox_heights[i],
                linewidth=0.3,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

        plt.xlim([0, self.image_width])
        plt.ylim([0, self.image_height])
        plt.title("BBoxes Distribution")
        plt.show()
        plt.savefig("plots/bbox_dist.png")
