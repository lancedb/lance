#!/usr/bin/env python3
#  Copyright (c) 2022. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import numpy as np


def iou(is_vectorized: bool, num_boxes: int = 100):
    if is_vectorized:
        return iou_vectorized(num_boxes)
    return iou_naive(num_boxes)


def iou_naive(num_boxes: int):
    xmin_arr = np.random.randn(num_boxes) + 1
    ymin_arr = np.random.randn(num_boxes) + 1
    xmax_arr = (np.random.randn(num_boxes) + 10) * 10
    ymax_arr = (np.random.randn(num_boxes) + 10) * 10
    ious = np.zeros((num_boxes, num_boxes))
    for i in range(num_boxes):
        for j in range(i, num_boxes):
            xmin = max(xmin_arr[i], xmin_arr[j])
            ymin = max(ymin_arr[i], ymin_arr[j])
            xmax = min(xmax_arr[i], xmax_arr[j])
            ymax = min(ymax_arr[i], ymax_arr[j])
            # compute the area of intersection rectangle
            inter = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
            # compute the area of both the prediction and ground-truth
            # rectangles
            area_i = ((xmax_arr[i] - xmin_arr[i] + 1) *
                      (ymax_arr[i] - ymin_arr[i] + 1))
            area_j = ((xmax_arr[j] - xmin_arr[j] + 1) *
                      (ymax_arr[j] - ymax_arr[j] + 1))
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            ious[i, j] = inter / float(area_i + area_j - inter)
            ious[j, i] = inter / float(area_i + area_j - inter)
    return ious


def iou_vectorized(num_boxes: int):
    xmin_arr = np.random.randn(num_boxes) + 1
    ymin_arr = np.random.randn(num_boxes) + 1
    xmax_arr = (np.random.randn(num_boxes) + 10) * 10
    ymax_arr = (np.random.randn(num_boxes) + 10) * 10
    area = (xmax_arr - xmin_arr + 1) * (ymax_arr - ymin_arr + 1)
    xmin_inter = np.maximum(xmin_arr, xmin_arr[:, np.newaxis])
    ymin_inter = np.maximum(ymin_arr, ymin_arr[:, np.newaxis])
    xmax_inter = np.minimum(xmax_arr, xmax_arr[:, np.newaxis])
    ymax_inter = np.minimum(ymax_arr, ymax_arr[:, np.newaxis])
    intersection = (np.max(xmax_inter - xmin_inter + 1, 0) *
                    np.max(ymax_inter - ymin_inter + 1, 0))
    union = area + area[:, np.newaxis] - intersection
    return intersection / union


if __name__ == "__main__":
    import time

    n_repeats = 10
    results = {}
    for num_boxes in [10, 100, 1000, 10000]:
        for is_vectorized in [True, False]:
            repeats = []
            for i in range(n_repeats):
                start = time.time_ns()
                iou(is_vectorized, num_boxes)
                end = time.time_ns()
                duration_ns = end - start
                repeats.append(duration_ns)
            results[(num_boxes, is_vectorized)] = np.mean(repeats)
    print(results)
