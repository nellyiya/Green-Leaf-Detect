import tensorflow as tf

def create_data_loaders(train_dir, test_dir, img_size=(128, 128), batch_size=32):
    """
    Creates TensorFlow datasets for training and validation.

    Args:
        train_dir (str): Path to training images directory.
        test_dir (str): Path to validation/testing images directory.
        img_size (tuple): Target image size (height, width).
        batch_size (int): Batch size.

    Returns:
        train_ds (tf.data.Dataset): Training dataset with augmentation and normalization.
        val_ds (tf.data.Dataset): Validation dataset with normalization only.
        class_names (list): List of class names in training dataset.
    """

    # Load datasets from directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Get class names BEFORE any map or prefetch operation
    class_names = train_ds.class_names

    # Data augmentation for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
    ])

    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    # Apply augmentation and normalization to training set
    def augment_and_normalize(image, label):
        image = data_augmentation(image)
        image = normalization_layer(image)
        return image, label

    train_ds = train_ds.map(augment_and_normalize)

    # Only normalize validation set
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


# If you want to test the script by running it standalone
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python preprocessing.py <train_dir> <test_dir>")
        sys.exit(1)

    train_dir = sys.argv[1]
    test_dir = sys.argv[2]

    train_ds, val_ds, class_names = create_data_loaders(train_dir, test_dir)

    print("Datasets created successfully.")
    print("Class names:", class_names)
    print("Number of batches in training set:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Number of batches in validation set:", tf.data.experimental.cardinality(val_ds).numpy())
