from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Optional

import cv2
import os
import glob

import numpy as np


class Process:
    def __init__(self):
        self.images_to_process: List[str] = []

        self.proces_flow: List[ProcessStage] = [ImageLoader(viz=True)]

    def read_image(self, image: str) -> None:
        self.images_to_process.append(os.path.abspath(image))

    def read_folder(self, folder_path: str) -> None:
        self.images_to_process.extend(glob.glob(os.path.join(folder_path, "*.jpg")))
        self.images_to_process.extend(glob.glob(os.path.join(folder_path, "*.png")))

    def process(self):
        for image in self.images_to_process:
            self.process_single(image)

    def process_single(self, image) -> bool:
        stage = image

        print("STARTING IMAGE PROCESSING", image)

        for process in self.proces_flow:
            print("STARTING", process.__class__.__name__)

            stage, success = process(stage)

            print("FINISHED", process.__class__.__name__, "SUCCESS", success)

            if not success:
                return False

        return True


class ProcessStage(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, input) -> Tuple[Any, bool]:
        pass

    def __call__(self, input):
        return self.process(input)


class ImageLoader(ProcessStage):
    def __init__(self, viz=False):
        super().__init__()
        self.viz = viz

    def process(self, input: str) -> Tuple[Optional[np.ndarray], bool]:
        if os.path.exists(input):
            img = cv2.imread(input)

            if self.viz:
                cv2.imshow("Image", img)
                cv2.waitKey()

            return img, True

        return None, False


if __name__ == '__main__':
    process = Process()

    process.read_folder('resources/raws')

    process.process()
