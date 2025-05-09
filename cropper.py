import numpy as np
import cv2
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import whisper

class Cropper():
    """
    model for cropping videos to just the lips/face
    """
    def __init__(self):
        self.lip_detector = None
        self.lip_tracker = None
        pass

    def extract_speaker_frame(self, video):
        """
        given a video returns a single frame in which the speaker is present for
        demographic/quality assessment
        """
        raise NotImplementedError

    def cut_lipless_frames(self, video, audio):
        """
        given a video and its audio, isolate the frames that actually have
        a face/lips in them (removes parts of videos that pan to a crowd or just
        dont have a face in them)

        returns: concatenated video/audio, saves to intermediate_videos
        """
        raise NotImplementedError

    def locate_lips(self, frame):
        """
        Find the locations of lips/face in a frame, draw a bounding box
        """
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)
        image = mp.Image.create_from_file("image.png")
        detection_result = detector.detect(image)

    def crop_to_lips(self, video, window_size = 255):
        """
        given a video, crops the video to a square, resizes
        to a standard size

        returns: cropped video + unchanged audio (since nothing new has been
        done to the audio), saves to processed_videos
        """
        raise NotImplementedError

    def create_transcript(self, audio, file_save):
        """
        Run whisper
        """
        model = whisper.load_model("base")
        audio = whisper.load_audio(audio)
        mel = whisper.log_mel_spectrogram(audio) #.to(model.device)
        options = whisper.DecodingOptions(language= 'en', fp16=False)
        result = whisper.decode(model, mel, options)
        with open(file_save, "w") as f:
            json.dump(result, f, indent=4) 
        print(result["text"])
        return result
    

if __name__ == "__main__":
    audio = "raw_videos/test_audio.mp3"
    crop = Cropper()
    crop.create_transcript(audio, "raw_videos/test_transcript.json")


