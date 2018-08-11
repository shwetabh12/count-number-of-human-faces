"""
smooth module

Image smoothers for better noise
handling
"""
import cv2

class Smoothers:
    """
    Methods for noise reduction
    """
    @classmethod
    def _best_smoother(cls, image):
        """
        Returns the best (and slower) smoother available
        """
        return cv2.bilateralFilter(image, d=5, sigmaSpace=200, sigmaColor=200)

    @classmethod
    def _fast_smoother(cls, image):
        """
        Implements fast smoothers for a more
        fluid detection
        """
        return cv2.blur(
            image,
            ksize=(5, 5)
        )
