import cv2
import numpy as np
from PIL import Image
import random

class LayoutAugmenter:
    def __init__(self):
        pass

    def augment(self, pil_image, max_objects=3):
        """
        Randomly shifts, scales, and multiplies the object 
        to create position and count diversity.
        """
        # Convert PIL to Numpy (OpenCV format)
        img = np.array(pil_image)
        
        # Handle grayscale/RGB conversion safely
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w, channels = img.shape

        # Create a blank black canvas
        canvas = np.zeros_like(img)
        
        # Decision: How many objects? (1 to max_objects)
        num_objects = random.randint(1, max_objects)
        
        for _ in range(num_objects):
            # 1. Random Scale (smaller if multiple objects)
            scale = random.uniform(0.5, 0.9) if num_objects > 1 else random.uniform(0.8, 1.1)
            
            # Resize
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            
            # 2. Random Position
            max_x = w - new_w
            max_y = h - new_h
            
            if max_x <= 0 or max_y <= 0: continue 
            
            start_x = random.randint(0, max_x)
            start_y = random.randint(0, max_y)
            
            # 3. Random Flip (Horizontal)
            if random.random() > 0.5:
                resized = cv2.flip(resized, 1)

            # 4. Paste logic (Max operator preserves white edges)
            roi = canvas[start_y:start_y+new_h, start_x:start_x+new_w]
            
            # If images are not same size (due to rounding), fix it
            if roi.shape[:2] != resized.shape[:2]:
                resized = cv2.resize(resized, (roi.shape[1], roi.shape[0]))

            combined = cv2.max(roi, resized)
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = combined

        return Image.fromarray(canvas)