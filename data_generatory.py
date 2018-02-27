import os
import random
import string

from argparse import ArgumentParser

from captcha.image import ImageCaptcha


class DataGenerator(object):
    """
    Class to generate the dataset. By default the dataset will be stored in the working directory in a separate folder.
    """
    def __init__(self):
        self.working_directory = os.getcwd()
        # This is where the dataset will be stored
        self.dataset_directory = os.path.join(self.working_directory, 'Dataset')

        # In case the dataset directory doesnt exist, create it
        if not os.path.isdir(self.dataset_directory):
            os.makedirs(self.dataset_directory)

    def generate_random_string(self, num_strings_to_generate, len_o_string=None):
        """
        Generates random strings of length specified.
        :param num_strings_to_generate: The number of strings to yield from this generator
        :param len_o_string: (optional) length of the string to generate. If not provided,
        variable length string will be used.
        :return:
        """
        if len_o_string is not None and len_o_string < 2:
            raise ValueError('To small string. Do you even want a string to be generated?')

        for _ in range(num_strings_to_generate):
            if len_o_string is None:

                # We want to generate a random string that contains both letters (lower and upper) and digits
                random_string = ''.join(
                    random.choice(
                        (string.ascii_letters + string.digits)
                    ) for _ in range(random.choice([4, 5, 6, 7, 8, 9, 10]))
                )

            else:
                random_string = ''.join(
                    random.choice(
                        (string.ascii_letters + string.digits)
                    ) for _ in range(len_o_string)
                )

            yield random_string


    def generate_dataset(self, num_samples, len_o_string=None):
        """
        Generates the dataset to use for training.
        :param num_samples: Number of samples to generate.
        :param len_o_string:(optional): length of the string to generate. If not provided, variable length string will
        be used.
        :return:
        """

        string_generator = self.generate_random_string(num_samples, len_o_string)

        image_captcha = ImageCaptcha()

        for random_string in string_generator:
            file_name = os.path.join(self.dataset_directory, '{}.png'.format(random_string))
            image_captcha.write(random_string, file_name)



if __name__ == '__main__':
    data_generator = DataGenerator()
    parser = ArgumentParser()

    parser.add_argument(
        '-n', '--num_samples', dest='num_samples', help='The number of samples to generate', required=True, type=int
    )

    parser.add_argument(
        '-l', '--length', dest='len_o_string', help='The length of the strings to generate', default=None, type=int
    )

    args = parser.parse_args()

    data_generator.generate_dataset(args.num_samples, args.len_o_string)
