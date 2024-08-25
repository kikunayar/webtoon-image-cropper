import cv2
import numpy as np
from pathlib import Path

def cropper(folder_path, out_path):
    folder_path = Path(folder_path)
    out_path = Path(out_path)
    
    # Create output directory if it does not exist
    out_path.mkdir(parents=True, exist_ok=True)

    # Get the list of files in the folder
    od = list(folder_path.glob('*.[jp][pn]g'))
    maskedimg = []
    fullsizeimg = []

    # Process each image file
    for img_path in od:
        if img_path.suffix.lower() in [".jpg", ".png"]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            fullsizeimg.append(img)
            h, w = img.shape[:2]
            img = cv2.resize(img, (w, h))
            
            # Apply masking
            for i in range(h):
                if np.all(img[i, :] == [255, 255, 255]):
                    img[i, :] = [255, 0, 0]  # Change white to red
                else:
                    img[i, :] = [0, 0, 255]  # Change non-white to blue
            img = cv2.resize(img, (1, h))
            maskedimg.append(img)

    if maskedimg:
        combined_maskedimg = np.vstack(maskedimg)
    if fullsizeimg:
        combined_fullsizeimg = np.vstack(fullsizeimg)

    image = combined_maskedimg
    pixels = image.reshape(-1, 3)  # Flatten the image and keep the color channels
    target_color = (255, 0, 0)  # Color we are interested in (red in BGR)
    to_color = (0, 0, 255)  # The color to change to (blue in BGR)

    def find_start_end(pixels, target_color):
        result = []
        start = None
        for i, value in enumerate(pixels):
            if tuple(value) == target_color:
                if start is None:
                    start = i
            else:
                if start is not None:
                    end = i - 1
                    length = end - start + 1
                    result.append((start, end, length))
                    start = None
        if start is not None:
            end = len(pixels) - 1
            length = end - start + 1
            result.append((start, end, length))
        return result

    def modify_image(image, start_end_ranges, min_length, to_color):
        modified_image = image.copy()
        pixels = modified_image.reshape(-1, 3)
        for start, end, length in start_end_ranges:
            if length < min_length:
                pixels[start:end + 1] = to_color
        return modified_image

    start_end_ranges = find_start_end(pixels, target_color)
    min_length = 100
    modified_image = modify_image(image, start_end_ranges, min_length, to_color)
    pixels = modified_image.reshape(-1, 3) 
    start_end_ranges = find_start_end(pixels, (0, 0, 255))

    if fullsizeimg:
        h, w = combined_fullsizeimg.shape[:2]

    # Save cropped images
    for idx, (start, end, _) in enumerate(start_end_ranges):
        cropped_img = combined_fullsizeimg[start:end + 1, :]  # Crop the image
        cropped_img_path = out_path / f"cropped_image_{idx}.png"
        cv2.imwrite(str(cropped_img_path), cropped_img)
        print(f"Cropped image saved to: {cropped_img_path}")


