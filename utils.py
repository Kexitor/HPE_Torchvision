import cv2
import matplotlib
import torch
import torchvision
import argparse
import utils
import time
from PIL import Image
from torchvision.transforms import transforms as transforms
import re
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# pairs of edges for 17 of the keypoints detected ...
# ... these show which points to be connected to which point ...
# ... we can omit any of the connecting points if we want, basically ...
# ... we can easily connect less than or equal to 17 pairs of points ...
# ... for keypoint RCNN, not  mandatory to join all 17 keypoint pairs
edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]


def draw_keypoints(outputs, image):
    ret_kps = []
    # the `outputs` is list which in-turn contains the dictionaries
    for i in range(len(outputs[0]['keypoints'])):
        keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
        # proceed to draw the lines if the confidence score is above 0.9
        if outputs[0]['scores'][i] > 0.9:
            keypoints = keypoints[:, :].reshape(-1, 3)
            ret_kps.append(keypoints)
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                            3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                # uncomment the following lines if you want to put keypoint number
                # cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie/float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb*255
                # join the keypoint pairs to draw the skeletal structure
                cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                        (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                        tuple(rgb), 2, lineType=cv2.LINE_AA)
        else:
            continue
    return image, ret_kps


def csv_converter(path, fname):
    df = pd.read_csv(path + fname, sep=";")
    df_ = pd.DataFrame(data=df)
    all_coordinates = []
    all_poses = []
    headers_names = df_.columns.values
    for col_num in range(len(df_)):
        all_poses.append([df_.loc[col_num].values[2]])
        all_coordinates.append([])
        for i in range(3, len(df_.loc[col_num].values)):
            nums = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", df_.loc[col_num].values[i])
            for num in nums:
                all_coordinates[col_num].append(float(num))

    return all_poses, all_coordinates


def pose_to_num(poses_):
    poses_list = ["walk", "fall", "fallen", "sitting"]
    all_poses_num = []
    for pose in poses_:
        if pose[0] == "walk":
            all_poses_num.append(["0"])
        if pose[0] == "fall":
            all_poses_num.append(["1"])
        if pose[0] == "fallen":
            all_poses_num.append(["2"])
        if pose[0] == "sitting":
            all_poses_num.append(["3"])

    return all_poses_num


def get_pose_from_num(pose_number):
    if pose_number[0] == "0":
        return "walk"
    if pose_number[0] == "1":
        return "fall"
    if pose_number[0] == "2":
        return "fallen"
    if pose_number[0] == "3":
        return "sitting"
    else:
        return "code_error"


def keypoints_parser(kps, dt_line):
    human = kps[0]
    for points in human:
        dt_line.append((round(points[0], 2), round(points[1], 2)))
    return dt_line


def get_coords_line(kps):
    coords_line = []
    for kp in kps:
        coords_line.append(kp[0])
        coords_line.append(kp[1])
    return coords_line