import random
import string

import numpy as np
from captcha.image import ImageCaptcha


class DataGenerator(object):
    """
    Class to generate the dataset. By default the dataset will be stored in the working directory in a separate folder.
    """

    def __init__(self, batch_size=32, length_of_captcha=4, height=80, width=170, num_channels=3):
        """
        Initiates a new instance of the DataGenerator class.
        :param batch_size: The number of samples to yield per call
        :param length_of_captcha: The number of characters in the captcha.
        :param height: height of the image to generate
        :param width: width of the image to generate
        :param num_channels: RGB = 3 BnW = 1
        """
        self.batch_size = batch_size
        self.length_of_capthca = length_of_captcha
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.characters = string.digits + string.ascii_uppercase
        self.n_class = len(self.characters)

    def generate_dataset(self):
        """
        Generates the dataset
        :return: a generator with the input and output with batch_size number of samples per call.
        """
        X = np.zeros((self.batch_size, self.num_channels, self.height, self.width), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.float32) for i in range(self.length_of_capthca)]

        while True:
            for i in range(self.batch_size):
                # Generate a random string
                random_str = ''.join([random.choice(self.characters) for j in range(self.length_of_capthca)])

                # The object that will generate our captcha
                generator = ImageCaptcha(width=self.width, height=self.height)
                image = generator.generate_image(random_str)

                # Convert the image into a nd array and in a shape that CNTK expects
                X[i] = np.asarray(image).reshape((self.num_channels, self.height, self.width))

                # One hot encode the output label
                for j, ch in enumerate(random_str):
                    y[j][i, :] = 0
                    y[j][i, self.characters.find(ch)] = 1
            yield X, y
