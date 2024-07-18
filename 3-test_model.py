import os
import tensorflow as tf
import numpy as np
import cv2
# import matplotlib.pyplot as plt

def preprocess_image(image, target_size=(28, 28)):
    """ Preprocess individual character image data """
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)  # Convert to (28, 28, 1)
    image = np.expand_dims(image, axis=0)   # Convert to (1, 28, 28, 1) for batch prediction
    return image

def segment_captcha(image_path, num_chars=4):
    """ Segment the captcha image into individual characters """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    img_height, img_width = binary_image.shape
    num_width = img_width // num_chars

    # plt.imshow(binary_image)
    # plt.show()

    characters = []
    for i in range(num_chars):
        left = i * num_width
        right = (i + 1) * num_width
        char_image = binary_image[0:img_height, left:right]
        characters.append(char_image)

    return characters

def predict_characters(model, characters):
    predictions = []
    for char_image in characters:
        preprocessed_image = preprocess_image(char_image)
        prediction = model.predict(preprocessed_image)
        predicted_label = np.argmax(prediction, axis=1)
        predictions.append(predicted_label[0])
    return predictions


def predict_captcha(model, image_path):
    characters = segment_captcha(image_path)
    predictions = predict_characters(model, characters)
    result = "".join(map(str, predictions))
    return result

def main():
    model = tf.keras.models.load_model("alexnet_model.h5")

    test_folder = "./test/"
    files = os.listdir(test_folder)
    for file in files:
        if not file.endswith(".png"):
            continue
        file_path = os.path.join(test_folder, file)
        image_name = os.path.splitext(file)[0]

        captcha_result = predict_captcha(model, file_path)
        print(f"Predicted: {captcha_result} <== {file}{'[✅]' if image_name == captcha_result else '[❌]'}")


if __name__ == "__main__":
    main()
