import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_to_tfexample(image_data, label):
    feature = {
        "image_raw": _bytes_feature(image_data),
        "label": _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def load_images_and_labels(folder_path):
    images = []
    labels = []
    label_mapping = {str(i): i for i in range(10)} # assuming digits 0-9

    for label_str, label_num in label_mapping.items():
        digit_folder_path = os.path.join(folder_path, label_str)
        if os.path.isdir(digit_folder_path):
            for file_name in os.listdir(digit_folder_path):
                if file_name.endswith(".png"):
                    file_path = os.path.join(digit_folder_path, file_name)
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    images.append(image)
                    labels.append(label_num)

    return images, labels

def write_tfrecords(images, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for image, label in zip(images, labels):
            image_data = image.tobytes()
            example = image_to_tfexample(image_data, label)
            writer.write(example.SerializeToString())

def main():
    images, labels = load_images_and_labels("processed")

    # Split into training and test sets
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # Generate TFRecord files
    os.makedirs("./dataset", exist_ok=True)
    write_tfrecords(images_train, labels_train, "./dataset/train.tfrecords")
    write_tfrecords(images_test, labels_test, "./dataset/test.tfrecords")

    print("TFRecord files generated successfully!")


if __name__ == "__main__":
    main()
