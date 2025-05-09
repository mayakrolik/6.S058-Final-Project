import numpy as np
import cv2
import torch
import torchaudio
import torchvision
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from face_detector import LandmarksDetector
from face_detector import VideoProcess
import os
import whisper

class Cropper(torch.nn.Module):
    """
    model for cropping videos to just the lips/face
    """
    def __init__(self, detector="retinaface"):
        super().__init__()
        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=False)

    def forward(self, data_filename):
        video = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        return video

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, frames_per_second)
    

if __name__ == "__main__":
    audio = "raw_videos/test_audio.mp3"
    cropper = Cropper("video")
    post_vid = cropper("raw_videos/raw_1_video.mp4")
    save2vid("./output.mp4", post_vid, frames_per_second=30)


