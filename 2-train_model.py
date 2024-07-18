import tensorflow as tf
# import tensorflowjs as tfjs

def _parse_function(proto):
    # Define the method for parsing features
    keys_to_features = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    # Parse the image data into a Tensor
    image = tf.io.decode_raw(parsed_features["image_raw"], tf.uint8)
    image = tf.reshape(image, [28, 28, 1])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(parsed_features["label"], tf.int32)
    return image, label

def load_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)
    return dataset

def build_alexnet(input_shape=(28, 28, 1), num_classes=10):
    return tf.keras.models.Sequential([
        # Specify the input shape
        tf.keras.layers.InputLayer(input_shape=input_shape),
        # First convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), strides=1),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        # Next convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        # Another convolutional layer
        tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        # Fully connected layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


def add_noise(image, label):
    # Add noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1, dtype=tf.float32)
    image = tf.add(image, noise)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def add_distort(image, label):
    transform_vector = [1, 0, tf.random.uniform([], -0.05, 0.05), 0, 1, tf.random.uniform([], -0.05, 0.05), 0, 0]
    transform_matrix = tf.reshape(transform_vector, [1, 8])

    image = tf.expand_dims(image, 0)
    transformed_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transform_matrix,
        output_shape=tf.shape(image)[1:3],
        interpolation='NEAREST',
        fill_value=0
    )
    transformed_image = tf.squeeze(transformed_image, 0)

    return transformed_image, label


def main():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

    origin_train_dataset = load_dataset("./dataset/train.tfrecords")
    augmented_train_dataset = origin_train_dataset.concatenate(origin_train_dataset.map(add_noise)).concatenate(origin_train_dataset.map(add_distort)).concatenate(origin_train_dataset.map(add_noise).map(add_distort))
    train_dataset = augmented_train_dataset.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = load_dataset("./dataset/test.tfrecords").batch(32).prefetch(tf.data.AUTOTUNE)

    alexnet_model = build_alexnet()
    alexnet_model.compile(optimizer="adam",
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        mode='min',
        restore_best_weights=True
    )

    alexnet_model.fit(
        train_dataset,
        epochs=50,
        validation_data=test_dataset,
        callbacks=[early_stopping])

    alexnet_model.save("alexnet_model.h5")
    # tfjs.converters.save_keras_model(alexnet_model, "tfjs_model")
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
