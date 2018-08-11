"""
classifiers

Image classifiers for object recognition
"""
import os
import cv2

class Classifiers:
    """
    Methods for image classification
    """
    _haar_features_file = os.path.dirname(
        os.path.realpath(
            __file__
        )
    ) + '/faces.xml'

    @classmethod
    def __init__(cls):
        cls._faces_haar_features = cv2.CascadeClassifier(cls._haar_features_file)

    @classmethod
    def _detect_face(cls, image):
        """
        Detect faces on image using Haar cascades
        """
        faces = cls._faces_haar_features.detectMultiScale(
            image,
            scaleFactor=2.5,
            minNeighbors=5,
            minSize=(30, 30),
        )

        return faces
