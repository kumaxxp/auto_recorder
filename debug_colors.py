
import cv2
import time
import numpy as np
from camera_stream import CameraStream
from collections import Counter

def main():
    print("Starting camera stream...")
    cam = CameraStream(
        index=0,
        width=640,
        height=480,
        use_gstreamer_segmentation=True,
        gstreamer_segmentation_config="./configs/deepstream_drivable_segmentation.txt"
    )
    
    cam.start()
    time.sleep(5) # Warmup
    
    print("Capturing...")
    mask = cam.read_mask()
    
    if mask is not None:
        # Reshape to list of pixels
        pixels = mask.reshape(-1, 3)
        # Convert to tuple to be hashable
        pixels_tuple = [tuple(p) for p in pixels]
        counts = Counter(pixels_tuple)
        
        total_pixels = mask.shape[0] * mask.shape[1]
        print(f"Total pixels: {total_pixels}")
        print("Color counts (BGR):")
        sorted_counts = counts.most_common()
        for color, count in sorted_counts:
            percentage = (count / total_pixels) * 100
            print(f"  Color {color}: {count} pixels ({percentage:.2f}%)")
            
        # The most common color is likely the background
        bg_color = sorted_counts[0][0]
        print(f"Likely background color: {bg_color}")
        
    else:
        print("Mask is None")

    cam.stop()

if __name__ == "__main__":
    main()
