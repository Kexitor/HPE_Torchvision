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
from data_lists import train_data, test_data
from utils import keypoints_parser


def get_mul_vid_data(data_path, markup):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', required=True,
    #                     help='path to the input data')
    # args = vars(parser.parse_args())

    # Transform to convert the image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Initialize the model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                   num_keypoints=17)
    # Set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the modle on to the computation device and set to eval mode
    model.to(device).eval()
    csvname = 'test_data_cap.csv'
    with open(csvname, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';')
        spamwriter.writerow(['time', "vname", 'pose', 'bp0', 'bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'bp6', 'bp7', 'bp8',
                             'bp9', 'bp10', 'bp11', 'bp12', 'bp13', 'bp14', 'bp15', 'bp16'])

    # Vid_counter can be changed differing on type of data (test up to 18 or train up to 30)
    # you are using from data_lists
    for video_n in range(0, 18):
        vid_path = data_path + markup[video_n][3]
        strange_falls = ["50wtf9.mp4", "50wtf12.mp4", "50wtf16.mp4",
                         "50wtf28.mp4", "50wtf31.mp4", "50wtf47.mp4", "50wtf49.mp4"]
        if markup[video_n][3] not in strange_falls:
            continue
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened() == False:
            print('Error while trying to read video. Please check path again')
        # Get the video frames' width and height
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # Set the save path
        # save_path = f"../outputs/{args['input'].split('/')[-1].split('.')[0]}.mp4"
        # # Define codec and create VideoWriter object
        # out = cv2.VideoWriter(save_path,
        #                       cv2.VideoWriter_fourcc(*'mp4v'), 20,
        #                       (frame_width, frame_height))
        frame_count = 0 # to count total frames
        total_fps = 0 # to get the final frames per second
        fps_time = 0
        frame_n = 0
        vid_fps = cap.get(cv2.CAP_PROP_FPS)

        # Read until end of video
        while (cap.isOpened()):
            # Capture each frame of the video
            ret, frame = cap.read()
            if ret == True:
                frame_number = frame_n / vid_fps
                frame_n += 1

                pil_image = Image.fromarray(frame).convert('RGB')
                orig_frame = frame
                data_line = []
                data_line.append(round(frame_number, 2))
                data_line.append(vid_path)

                # Video frames
                time_1 = markup[video_n][1]
                time_2 = markup[video_n][2]
                init_pose = markup[video_n][0] # "sitting" # "walk"
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
                # Getting human coordinates
                try:
                    data_line = keypoints_parser(keypoints_, data_line)
                except:
                    pass

                # Get the fps
                fps = 1 / (end_time - start_time)
                # Add fps to total fps
                total_fps += fps
                # Increment frame count
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
                    with open(csvname, 'a', newline='') as csvfile:
                        spamwriter = csv.writer(csvfile, delimiter=';')
                        spamwriter.writerow(data_line)
                # out.write(output_image)
                # Press `q` to exit
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release VideoCapture()
        cap.release()
        # Close all frames and video windows
        cv2.destroyAllWindows()
        # Calculate and print the average FPS
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


data_path_ = "videos/cuts_test/"
get_mul_vid_data(data_path_, test_data)
