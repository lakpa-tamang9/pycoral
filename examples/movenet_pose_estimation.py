# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to estimate a single human pose with Edge TPU MoveNet.

To run this code, you must attach an Edge TPU to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

For more details about MoveNet and its best practices, please see
https://www.tensorflow.org/hub/tutorials/movenet

Example usage:
```
bash examples/install_requirements.sh movenet_pose_estimation.py

python3 examples/movenet_pose_estimation.py \
  --model test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite  \
  --input test_data/squat.bmp
```
"""

import argparse

from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import cv2
import numpy as np
import tensorflow as tf

_NUM_KEYPOINTS = 17

EDGES = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--model", required=True, help="File path of .tflite file."
    )
    args = parser.parse_args()

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        # Reshape image
        img = frame.copy()
        resized_img = img.resize(common.input_size(interpreter), Image.ANTIALIAS)
        common.set_input(interpreter, resized_img)
        output_details = interpreter.get_output_details()
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])

        pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)

        # Rendering
        draw_connections(img, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(img, keypoints_with_scores, 0.4)

        cv2.imshow("MoveNet Lightning", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
    # print(pose)
    # draw = ImageDraw.Draw(img)
    # width, height = img.size
    # for i in range(0, _NUM_KEYPOINTS):
    #     draw.ellipse(
    #         xy=[
    #             pose[i][1] * width - 2,
    #             pose[i][0] * height - 2,
    #             pose[i][1] * width + 2,
    #             pose[i][0] * height + 2,
    #         ],
    #         fill=(255, 0, 0),
    #     )
    # img.save(args.output)
    print("Done. Results saved at", args.output)


if __name__ == "__main__":
    main()
