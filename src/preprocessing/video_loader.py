"""Video loading and preprocessing utilities."""
import cv2
from pathlib import Path


class VideoLoader:
    """Load and preprocess video files for model input."""
    
    def __init__(self, target_size=(224, 224)):
        """Initialize video loader.
        
        Args:
            target_size: Tuple of (height, width) for resizing frames.
        """
        self.target_size = target_size
    
    def load_video(self, video_path):
        """Load video from path and extract frames.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            List of preprocessed frames.
        """
        # TODO: Implement video loading
        pass
    
    def preprocess_frame(self, frame):
        """Preprocess individual frame.
        
        Args:
            frame: Raw video frame as numpy array.
            
        Returns:
            Preprocessed frame ready for model input.
        """
        # TODO: Implement preprocessing
        pass
