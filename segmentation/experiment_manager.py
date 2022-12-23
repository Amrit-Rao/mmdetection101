from __future__ import annotations
from mmdet.datasets import build_dataset, build_dataloader, CocoDataset
from mmcv import dump
from mmcv.runner import load_checkpoint
from mmdet.apis import (
    train_detector,
    set_random_seed,
    single_gpu_test,
)
from mmdet.models import build_detector
from mmdet.utils import build_dp
from pydantic import BaseModel
from typing import Union, Dict, Any, List, Optional, Tuple
from pathlib import Path
from mmcv import Config
import datetime
import os


primitive_type = Union[int, float, str, bool]
file_path = Path(__file__)


class ExperimentManager(BaseModel):
    name: str = str(datetime.datetime.now())
    model: Optional[Any]
    train_dataset: Optional[Any]
    train_dataloader: Optional[Any]
    val_dataset: Optional[Any]
    val_dataloader: Optional[Any]
    test_dataset: Optional[Any]
    test_dataloader: Optional[Any]

    config: Optional[
        Dict[
            str, Union[primitive_type, List[primitive_type], Dict[str, primitive_type]]
        ]
    ]

    def getValDataloader(self):

        return self.val_dataloader

    def getModel(self):

        return self.model

    def getConfigFromPath(
        self,
        *,
        config_path: Union[str, Path],
    ) -> ExperimentManager:
        # Set model config parameters from file
        self.config = Config.fromfile(config_path)
        return self

    def setConfigToPath(
        self, *, config_path: Optional[Union[str, Path]] = None
    ) -> ExperimentManager:
        # Store model config parameters to specified path
        if config_path is None:
            filename = self.name + ".py"
        elif os.path.exists(config_path):
            filename = config_path + "/" + self.name + ".py"
        else:
            os.makedirs(config_path)
            filename = config_path + "/" + self.name + ".py"

        with open(filename, "w") as f:
            f.write(str(dict(self.config)) + "\n")
        return self

    def buildDataset(
        self,
        *,
        data_path: Union[str, Path],
        type: str = "CocoDataset",
        classes: Tuple = CocoDataset.CLASSES,
        workers_per_gpu: int = 1,
        samples_per_gpu: int = 1,
        number_of_gpus: int = 1,
        device="cuda",
    ):
        # Build Dataset based on config file

        self.config.gpu_ids = range(number_of_gpus)
        self.config.device = device

        self.config.data.train.type = type
        self.config.data.train.ann_file = data_path + "/train/labels.json"
        self.config.data.train.img_prefix = data_path + "/train/data/"
        self.config.data.train.classes = classes

        self.config.data.val.type = type
        self.config.data.val.ann_file = data_path + "/validation/labels.json"
        self.config.data.val.img_prefix = data_path + "/validation/data/"
        self.config.data.val.classes = classes

        self.config.data.test.type = type
        self.config.data.test.ann_file = data_path + "/test/labels.json"
        self.config.data.test.img_prefix = data_path + "/test/data/"
        self.config.data.test.classes = classes

        self.train_dataset = build_dataset(self.config.data.train)
        self.val_dataset = build_dataset(self.config.data.val, dict(test_mode=True))
        self.test_dataset = build_dataset(self.config.data.test, dict(test_mode=True))

        self.train_dataloader = build_dataloader(
            self.train_dataset,
            samples_per_gpu,
            workers_per_gpu,
            number_of_gpus,
        )
        self.val_dataloader = build_dataloader(
            self.val_dataset,
            samples_per_gpu,
            workers_per_gpu,
            number_of_gpus,
            shuffle=False,
        )
        self.test_dataloader = build_dataloader(
            self.test_dataset,
            samples_per_gpu,
            workers_per_gpu,
            number_of_gpus,
            shuffle=False,
        )

        return self

    def buildModel(
        self,
        *,
        classes: tuple = CocoDataset.CLASSES,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> ExperimentManager:
        # Build Model on specified config file
        self.config.model.roi_head.bbox_head.num_classes = len(classes)
        try:
            self.config.model.roi_head.mask_head.num_classes = len(classes)
        except AttributeError:
            print("No Mask in Model configuration. Continuing without masks")

        self.model = build_detector(dict(self.config.model))
        if checkpoint_path is not None:
            checkpoint = load_checkpoint(
                self.model, checkpoint_path, map_location=self.config.device
            )
            if "CLASSES" in checkpoint.get("meta", {}):
                self.model.CLASSES = checkpoint["meta"]["CLASSES"]
            else:
                print(
                    "No classes detected in checkpoint file. Using given classes instead"
                )
                self.model.CLASSES = classes
        self.model.cfg = self.config
        return self

    def train(
        self,
        *,
        output_path: Optional[Union[str, Path]],
        num_of_epochs: int = 1,
        eval_interval: int = 2,
        checkpoint_interval: int = 2,
        metric: str = "bbox",
    ) -> ExperimentManager:
        # Train on model
        self.config.seed = 0
        set_random_seed(0, deterministic=False)
        self.config.optimizer.lr = 0.02 / 8
        self.config.lr_config.warmup = None
        self.config.log_config.interval = 10
        self.config.runner.max_epochs = num_of_epochs
        self.config.evaluation.metric = metric
        self.config.evaluation.interval = eval_interval
        self.config.checkpoint_config.interval = checkpoint_interval
        if output_path is None:
            self.config["work_dir"] = self.name
        else:
            self.config["work_dir"] = str(output_path) + "/" + self.name

        train_detector(
            self.model,
            self.train_dataset,
            self.config,
            distributed=False,
            validate=True,
        )
        return self

    def predict(
        self,
        *,
        output_path: Optional[Union[str, Path]],
        show_output: bool = True,
        score_threshold: float = 0.3,
    ) -> ExperimentManager:
        model = build_dp(self.model, self.config.device, device_ids=self.config.gpu_ids)
        outputs = single_gpu_test(
            model,
            self.test_dataloader,
            show_output,
            output_path,
            score_threshold,
        )
        print(outputs)

        return self

    def evaluate(
        self,
        *,
        output_path: Optional[Union[str, Path]],
        show_output: bool = True,
        score_threshold: float = 0.3,
    ) -> ExperimentManager:

        # Evaluate on dataset from path
        model = build_dp(self.model, self.config.device, device_ids=self.config.gpu_ids)
        outputs = single_gpu_test(
            model,
            self.val_dataloader,
            show_output,
            output_path,
            score_threshold,
        )
        results = self.val_dataloader.dataset.evaluate(outputs)

        return self
