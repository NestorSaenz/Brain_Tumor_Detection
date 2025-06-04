
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

class DataGenerator:
    def __init__(self, batch_size, img_size):
        """
        Initializes the data generator with batch size and image size.
        Args:
            batch_size (int): Number of samples per batch.
            img_size (tuple): Size of the images (height, width).
        """
        self.batch_size = batch_size
        self.img_size = img_size

    def get_train_datagen(self):
        return ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def get_test_datagen(self):
        return ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    def get_generator(self, df, datagen, shuffle=True):
        return datagen.flow_from_dataframe(
            df,
            x_col='image_path',
            y_col='category_encoded',
            target_size=self.img_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=shuffle,
            batch_size=self.batch_size
        )
