# This file was copied from https://github.com/gleporoni/rgbd-depthfake/blob/main/src/data/faceforensics.py
# Commit id: 992930f1b28d1fc80e73307aa86388eccd3b8d5d

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from typing import Any, Tuple
from pathlib import Path
from omegaconf import DictConfig
from PIL import Image

from torch.utils.data import Dataset
import os

from data.const import VAL_VIDEOS, TEST_VIDEOS


logger = logging.getLogger(__name__)


class FaceForensics(Dataset):
    """
    Dataset loader for T4SA dataset.
    """

    def __init__(
        self,
        conf: DictConfig,
        split: str,
        transform: Any = None,
    ):
        self.conf = conf
        self.split = split
        self.num_classes = self.conf.data.num_classes

        # Data dirs
        self.base_path = Path(Path(__file__).parent, "../../")
        self.rgb_path = Path(self.base_path, self.conf.data.rgb_path)
        self.depth_path = Path(self.base_path, self.conf.data.depth_path)

        # Dataset info
        self.compression_level = self.conf.data.compression_level
        self.real = self.conf.data.real
        self.attacks = self.conf.data.attacks
        self.depth_type = self.conf.data.depth_type
        self.use_depth = self.conf.data.use_depth
        self.input_type = self.conf.data.input_type

        # Val and test data
        self.val_videos = VAL_VIDEOS
        self.test_videos = TEST_VIDEOS

        self.dataset = self._load_data(
            use_attacks=self.conf.data.use_attacks, use_depth=self.use_depth
        )

        # Convert string labels to categoricals
        self._to_categorical()

        # Split the dataset
        self._data_split()

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        This function returns a tuple that is further passed to collate_fn
        """
        # Load the image and apply transformations
        image = Image.open(self.dataset.images[idx]).convert("RGB")

        if self.input_type == "rgbd":
            # Load depth
            if self.depth_type == "hha":
                image = np.array(image)
                tmp = str(self.dataset.depths[idx]).split("Depth-Faces")
                path = tmp[0] + "HHA-Faces" + tmp[1].split(".")[0] + ".png"
                hha = Image.open(path).convert("RGB")
                hha = np.array(hha)

                image = np.stack(
                    (
                        image[:, :, 0],
                        image[:, :, 1],
                        image[:, :, 2],
                        hha[:, :, 0],
                        hha[:, :, 1],
                        hha[:, :, 2],
                    ),
                    axis=-1,
                )

            elif self.depth_type == "depth":
                depth = np.load(self.dataset.depths[idx], allow_pickle=True)

                # Normalize depth to [0, 1] per image
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

                image = np.array(image)
                image = np.stack(
                    (image[:, :, 0], image[:, :, 1], image[:, :, 2], depth), axis=-1
                )
            else:
                raise NotImplementedError

        elif self.input_type == "d":
            # Load depth
            if self.depth_type == "hha":
                tmp = str(self.dataset.depths[idx]).split("Depth-Faces")
                path = tmp[0] + "HHA-Faces" + tmp[1].split(".")[0] + ".png"
                hha = Image.open(path).convert("RGB")
                image = np.array(hha)

            elif self.depth_type == "depth":
                image = np.load(self.dataset.depths[idx], allow_pickle=True)
            else:
                raise NotImplementedError

        elif self.input_type == "rgb":
            pass
            # do nothing
        else:
            raise NotImplementedError

        if self.transform:
            image = self.transform(image)
        label = self.dataset.classes[idx]

        return {
            "image": image,
            "label": label,
        }

    def _load_data(
        self, use_depth: bool = False, use_attacks: list = False
    ) -> pd.DataFrame:
        """
        Load the RGB images.
        """
        images = []
        depths = []
        labels = []

        # Loop over compression levels
        for compression in self.compression_level:
            # Loop over real videos
            for real in self.real:
                if use_depth:
                    # Add depths
                    list_of_depths = self._load_depth(
                        compression=compression, label="Real", source=real
                    )

                    # Add RGB images
                    list_of_images = self._load_rgb_from_depth(
                        compression=compression,
                        depths=list_of_depths,
                        label="Real",
                        class_id=real,
                    )
                    list_of_images, list_of_depths = self._validate_depths(
                        list_of_images
                    )

                    # Add labels
                    list_of_labels = self._load_labels(list_of_images, real)

                    images += list_of_images
                    depths += list_of_depths
                    labels += list_of_labels
                else:
                    # Add RGB images
                    list_of_images = self._load_rgb(
                        compression, label="Real", class_id=real
                    )
                    images += list_of_images

                    # Add depths
                    for _ in range(len(list_of_images)):
                        depths.append(None)

                    # Add labels
                    list_of_labels = self._load_labels(list_of_images, real)
                    labels += list_of_labels

            if use_attacks:
                # Loop over the attacks
                for attack in self.attacks:
                    if use_depth:
                        # Add depths
                        list_of_depths = self._load_depth(
                            compression=compression, label="Fake", source=attack
                        )

                        # Add RGB images
                        list_of_images = self._load_rgb_from_depth(
                            compression=compression,
                            depths=list_of_depths,
                            label="Fake",
                            class_id=attack,
                        )
                        list_of_images, list_of_depths = self._validate_depths(
                            list_of_images
                        )

                        # Add labels
                        list_of_labels = self._load_labels(list_of_images, attack)

                        images += list_of_images
                        depths += list_of_depths
                        labels += list_of_labels
                    else:
                        # Add RGB images
                        list_of_images = self._load_rgb(
                            compression, label="Fake", class_id=attack
                        )
                        images += list_of_images

                        # Add depths
                        for _ in range(len(list_of_images)):
                            depths.append(None)

                        # Add labels
                        list_of_labels = self._load_labels(list_of_images, attack)

                        labels += list_of_labels

        dataset = pd.DataFrame(
            data={
                "images": images,
                "depths": depths,
                "labels": labels,
            }
        )

        return dataset

    def _load_rgb(self, compression, label, class_id):
        images = [
            path
            for path in Path(self.rgb_path, label, compression, class_id).glob(
                "*/*.jpg"
            )
        ]
        logger.info(
            f"Found {len(images)} images for params label={label} compression={compression} class_id={class_id}"
        )
        return images

    def _load_labels(self, images, class_id):
        labels = [class_id for _ in range(len(images))]

        return labels

    def _load_depth(self, label: str, compression: str, source: str) -> list:
        assert (
            label in ("Real", "Fake")
            and source
            in (
                "actors",
                "youtube",
                "Deepfakes",
                "Face2Face",
                "FaceShifter",
                "FaceSwap",
                "NeuralTextures",
            )
            and compression in ("raw", "c23", "c40")
        )

        depths = [
            path
            for path in Path(self.depth_path, label, compression, source).glob(
                "*/*.npy"
            )
            if not self._is_file_empty(path)
        ]

        logger.info(
            f"Found {len(depths)} depth maps for params label={label} compression={compression} source={source}"
        )

        return depths

    def _load_rgb_from_depth(self, compression, depths, label, class_id):
        """
        Converts a list of depth file paths to their corresponding RGB image paths,
        then filters to keep only those RGB paths that actually exist.
        """

        depth_images = []
        for depth in depths:
            rgb_path = str(depth).replace(
                self.conf.data.depth_path, self.conf.data.rgb_path
            )
            rgb_path = rgb_path.replace("_d.npy", ".jpg")
            depth_images.append(Path(rgb_path))

        # Remove paths if they do not exist
        rgb_images = self._load_rgb(compression, label=label, class_id=class_id)
        images_to_remove = set(depth_images) - set(rgb_images)
        images = list(set(depth_images) - images_to_remove)

        return images

    def _validate_depths(self, images):
        """
        Removes the paths that do not exist in the dataset.
        """
        # Remove depths that do not have a correspondance with a rgb path
        new_depths = []
        for image in images:
            depth_path = str(image).replace(
                self.conf.data.rgb_path, self.conf.data.depth_path
            )
            depth_path = depth_path.replace(".jpg", "_d.npy")
            new_depths.append(Path(depth_path))

        return images, new_depths

    def _to_categorical(self):
        """
        Converts strings labels to integers.
        """
        classes = []
        for label in self.dataset.labels:
            if label in self.conf.data.real:
                classes.append(0)
            else:
                if self.num_classes == 2:
                    classes.append(1)
                else:
                    classes.append(list(self.conf.data.attacks).index(label) + 1)

        self.dataset["classes"] = classes

    def _data_split(self):
        """
        Splits the dataset according to the actual split.
        """
        split = []
        for _, row in self.dataset.iterrows():
            video = str(row.images.parent).split("/")[-1].split("_")[0]
            if video in self.val_videos:
                split.append("val")
            elif video in self.test_videos:
                split.append("test")
            else:
                split.append("train")
        self.dataset["split"] = split

        self.dataset = self.dataset.loc[
            self.dataset["split"] == self.split
        ].reset_index()

    def _is_file_empty(self, file_path):
        return os.stat(file_path).st_size == 0
