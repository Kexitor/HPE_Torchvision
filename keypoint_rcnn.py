import torch
import torchvision
import cv2
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
import pickle
from utils import csv_converter, pose_to_num, get_pose_from_num, get_coords_line

#  python keypoint_rcnn.py --input videos\50wtf.mp4

# Getting train dataset
path = ""  # "videos/csv_files/"
filename = "37vid_data.csv"
train_poses, train_coords = csv_converter(path, filename)
train_poses_num = pose_to_num(train_poses)

# Training model

# NN = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1,
#                    max_iter=10000).fit(train_coords, train_poses_num)
# RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(train_coords, train_poses_num)
# LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train_coords, train_poses_num)

# pkl_filename = "pm_37vtrain_tv.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(NN, file)
NN = ""
with open("pm_37vtrain_tv.pkl", 'rb') as file:
    NN = pickle.load(file)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True,
                    help='path to the input data')
args = vars(parser.parse_args())
# Transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
# Initialize the model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                               num_keypoints=17)
# Set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
# Load the model on to the computation device and set to eval mode
model.to(device).eval()

cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the video frames' width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Set the save path
# save_path = f"../outputs/{args['input'].split('/')[-1].split('.')[0]}.mp4"
# # Define codec and create VideoWriter object
# out = cv2.VideoWriter(save_path,
#                       cv2.VideoWriter_fourcc(*'mp4v'), 20,
#                       (frame_width, frame_height))

# VideoWriter for saving the video
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter("50wtf_torch.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second

fps_time = 0
pose_label = "none"
# Read until end of video
while cap.isOpened():
    # Capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        pil_image = Image.fromarray(frame).convert('RGB')
        orig_frame = frame
        # Transform the image
        image = transform(pil_image)
        # Add a batch dimension
        image = image.unsqueeze(0).to(device)
        # Get the start time
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image)
        # Get the end time
        end_time = time.time()
        output_image, keypoints_ = utils.draw_keypoints(outputs, orig_frame)
        # Get the fps
        fps = 1 / (end_time - start_time)
        # Add fps to total fps
        total_fps += fps
        # Increment frame count
        frame_count += 1
        wait_time = max(1, int(fps / 4))

        # Classifying pose for identified human, can classify several humans poses in frame
        coords_line = []
        try:
            coords_line = [get_coords_line(keypoints_[0])]
            for human_kps in keypoints_:
                hum_crd_ln = [get_coords_line(human_kps)]
                if 36 >= len(hum_crd_ln) >= 1:
                    pose_code = NN.predict(hum_crd_ln)
                    pose_label = get_pose_from_num(pose_code)
                    cv2.putText(output_image,
                                "pose: %s" % (pose_label),
                                (int(hum_crd_ln[0][0]), int(hum_crd_ln[0][1]) - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)
        except:
            pass
        pose_label = "none"
        if 34 >= len(coords_line) >= 1:
            pose_code = NN.predict(coords_line)
            pose_label = get_pose_from_num(pose_code)

        cv2.putText(output_image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(output_image,
                    "NN: %s" % (pose_label),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.imshow('Pose detection frame', output_image)
        fps_time = time.time()
        # out.write(output_image)
        # press `q` to exit
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
# out.release()
cap.release()
# Close all frames and video windows
cv2.destroyAllWindows()
# Calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
