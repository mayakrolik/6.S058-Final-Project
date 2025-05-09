import numpy as np
import cv2
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import models.predict
import whisper

class Classifier():
    """
    Handles methods related to determining the desired demographic information:

    - race
    - age
    - gender
    
    as well as some non-demographic lables:
    - face orientation
    - lighting
    - resolution
    """
    def __init__(self):
        pass

    def race_age_gender_classification(self, csv_of_img_paths):
        """
        Runs FairFace
        """
        models.predict(csv_of_img_paths)


