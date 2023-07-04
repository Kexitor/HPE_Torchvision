import torch
import torchvision
import cv2
import argparse
import utils
import time
import csv
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms as transforms


def keypoints_parser(kps, dt_line):
    human = kps[0]
    for points in human:
        dt_line.append((round(points[0], 2), round(points[1], 2)))
    return dt_line

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True,
                    help='path to the input data')
args = vars(parser.parse_args())
# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
# initialize the model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                               num_keypoints=17)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.is_available())
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the video frames' width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# set the save path
# save_path = f"../outputs/{args['input'].split('/')[-1].split('.')[0]}.mp4"
# # define codec and create VideoWriter object
# out = cv2.VideoWriter(save_path,
#                       cv2.VideoWriter_fourcc(*'mp4v'), 20,
#                       (frame_width, frame_height))
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

fps_time = 0

frame_n = 0
vid_fps = cap.get(cv2.CAP_PROP_FPS)

csvname = args['input'] + '.csv'
with open(csvname, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';')
    spamwriter.writerow(['time', "vname", 'pose', 'bp0', 'bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'bp6', 'bp7', 'bp8',
                         'bp9', 'bp10', 'bp11', 'bp12', 'bp13', 'bp14', 'bp15', 'bp16'])

# read until end of video

video_data_coords = []
while (cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        frame_number = frame_n / vid_fps
        frame_n += 1

        pil_image = Image.fromarray(frame).convert('RGB')
        orig_frame = frame
        data_line = []
        data_line.append(round(frame_number, 2))
        data_line.append(args['input'])

        # video frames
        time_1 = 1.0
        time_2 = 1.6
        init_pose = "walk" # "sitting" # "walk"
        pose_label = "none"
        if frame_number < time_1:
            pose_label = init_pose
        if time_1 <= frame_number < time_2:
            pose_label = "fall"
        if frame_number >= time_2:
            pose_label = "fallen"
        data_line.append(pose_label)
        print("#####")
        print(data_line)
        print(frame_n)
        print(vid_fps)
        print("#####")
        # transform the image
        image = transform(pil_image)
        # add a batch dimension
        image = image.unsqueeze(0).to(device)
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image)
        # get the end time
        end_time = time.time()
        output_image, keypoints_ = utils.draw_keypoints(outputs, orig_frame)
        human_kps = []
        try:
            data_line = keypoints_parser(keypoints_, data_line)
        except:
            pass
        # data_line.append(human_kps)
        # get the fps
        # for cntr in range(len(keypoints_)):
        #     print("human num", cntr)
        #     print(keypoints_[cntr])
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        wait_time = max(1, int(fps / 4))
        cv2.putText(output_image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(output_image,
                    pose_label,
                    (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('Pose detection frame', output_image)
        fps_time = time.time()
        if 23 > len(data_line) > 3:
            # data_line[2] = "none"
            with open(csvname, 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=';')
                spamwriter.writerow(data_line)
        # out.write(output_image)
        # press `q` to exit
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")

