import cv2
import numpy as np
import os
import tqdm

# Load the image and its corresponding mask

def crop_image(image,mask):
    # Convert the mask to binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Assuming there is only one contour (the main object), find its bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the crop region around the center
    crop_size = max(w,h)
    crop_x1 = max(0, center_x - crop_size // 2)
    crop_y1 = max(0, center_y - crop_size // 2)
    crop_x2 = min(image.shape[1], crop_x1 + crop_size)
    crop_y2 = min(image.shape[0], crop_y1 + crop_size)


    rgb_float = image.astype(np.float32) / 255.0
    mask_float = mask.astype(np.float32) / 255.0

        # Multiply the image with the mask
    result = (rgb_float * mask_float[:, :, np.newaxis])

        # Create a white background
    white_background = np.ones_like(rgb_float)  # All ones for white
    white_background *= (1 - mask_float[:, :, np.newaxis])  # Multiply by inverse mask

        # Add the content and the white background
    result += white_background
    

        # Convert result back to uint8
    result = (result * 255).astype(np.uint8)
    # Crop the image
    cropped_image = result[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_image = cv2.resize(cropped_image, (512, 512), interpolation=cv2.INTER_AREA)

    return cropped_image
# Display the cropped image



# Example usage
if __name__ == "__main__":
    # Load RGB image and mask

    section = "train"
    obj = "youtube5"#"youtube_bee"

    out_path = f'/GaussianAvatar_normal/data/{obj}/{section}/cropped_images'

    if os.path.exists(out_path) is False:
        os.mkdir(out_path)
    
    image_path = f'/GaussianAvatar_normal/data/{obj}/{section}/images'
    mask_path =f'GaussianAvatar_normal/data/{obj}/{section}/masks'

    img_files = os.listdir(image_path)
    img_files = sorted(img_files)

    for img in img_files:
        image = cv2.imread(os.path.join(image_path,img))
        mask = cv2.imread(os.path.join(mask_path,img), cv2.IMREAD_GRAYSCALE)



    # Remove background
        result_image = crop_image(image, mask)

    # Display the result

        cv2.imwrite(os.path.join(out_path,img),
                   
                    result_image)
 