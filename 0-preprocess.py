import os
import cv2
# import numpy as np

def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarization processing (thresholding)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    return binary_image

    # Use morphological operations to remove noise
    # kernel = np.ones((2,2), np.uint8)
    # cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    # return cleaned_image

def process_captcha(image_path, label, target_folder):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    processed_image = preprocess_image(image)
    img_height, img_width = processed_image.shape
    num_width = img_width // len(label)

    for i, contour in enumerate(label):
        # Define the cropping region for each digit
        left = i * num_width
        right = (i + 1) * num_width
        digit_image = processed_image[0:img_height, left:right]

        # Resize - Adjust to 28x28 pixels
        resized_digit = cv2.resize(digit_image, (28, 28), interpolation=cv2.INTER_AREA)

        # Define the target folder path
        folder_path = os.path.join(target_folder, contour)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        # Define the file path to ensure unique file names
        file_index = 1
        file_path = os.path.join(folder_path, f"{file_index}.png")
        while os.path.exists(file_path):
            file_index += 1
            file_path = os.path.join(folder_path, f"{file_index}.png")
        cv2.imwrite(file_path, resized_digit)

def main():
    data_folder = "data"

    for file_name in os.listdir(data_folder):
        if file_name.endswith(".png"):
            image_name = os.path.splitext(file_name)[0]
            if "_" in image_name:
                image_name = image_name.split("_")[0]
            image_path = os.path.join(data_folder, file_name)
            print(f"Processing file: {image_path} -> {image_name}")
            process_captcha(image_path, image_name, "processed")

    print("All images preprocessed and saved successfully!")

if __name__ == "__main__":
    main()
