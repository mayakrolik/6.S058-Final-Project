import numpy as np
import cv2

class Cropper():
    """
    model for cropping videos to just the lips/face
    """
    def __init__(self):
        self.lip_detector = None
        self.lip_tracker = None
        pass

    def cut_lipless_frames(self, video, audio):
        """
        given a video and its audio, isolate the frames that actually have
        a face/lips in them (removes parts of videos that pan to a crowd or just
        dont have a face in them)

        returns: concatenated video/audio, saves to intermediate_videos
        """
        pass

    def crop_to_lips(self, video, window_size = 255):
        """
        given a video, crops the video to a square, resizes
        to a standard size

        returns: cropped video + unchanged audio (since nothing new has been
        done to the audio), saves to processed_videos
        """
        pass

