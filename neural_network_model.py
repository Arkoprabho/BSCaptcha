"""
Module that defines the neural network model.
"""
from _cntk_py import set_fixed_random_seed

import cntk
import numpy as np
from tqdm import tqdm

from data_handler.data_generator import DataGenerator


class NeuralNetworkModel(object):
    """
    The model of the neural network.
    The model is inspired from the VGG architecture.
    The architecture has a shared convolution base with branched Dense layers at the end to predict the final outcome.
    Each dense layer is responsible for each of the characters in the captcha.
    """

    def __init__(self, shape, num_class, length_of_captcha=4, seed=0):
        """
        Initiates a new instance of the neural network model class
        :param length_of_captcha: The length of captcha being used. This will severely determine the structure of the
        neural network. The number of Dense branches and the number of basic convolutional modules
        depend on this parameter.
        :param seed: The random seed to use. This will change both the numpy and the CNTK seed.
        """
        self.num_dense_branches = length_of_captcha
        self.num_convolution_layers = length_of_captcha
        self.seed = seed
        self.shape = shape
        self.num_class = num_class

        np.random.seed(seed)
        set_fixed_random_seed(seed)

    def create_model(self, features):
        """
        Creates the model that will predict the captchas.
        :param features: the input image to be fed
        :param num_class: the number of classes the output variable can have
        :return:
        """
        inner_features = features
        # The convolutional base
        with cntk.layers.default_options(init=cntk.glorot_uniform(), activation=cntk.relu):
            for layer_number in range(self.num_convolution_layers):
                # 2 Conv 1 MaxPool repeated 4 times
                inner_features = cntk.layers.Convolution2D(
                    filter_shape=(3, 3),
                    num_filters=32 * 2 ** layer_number,
                    name='Conv1 {}'.format(layer_number + 1)
                )(inner_features)

                inner_features = cntk.layers.Convolution2D(
                    filter_shape=(3, 3),
                    num_filters=32 * 2 ** layer_number,
                    name='Conv2 {}'.format(layer_number + 1)
                )(inner_features)

                inner_features = cntk.layers.MaxPooling(
                    filter_shape=(2, 2),
                    name='MaxPool {}'.format(layer_number + 1)
                )(inner_features)

            # Prevent overfitting
            inner_features = cntk.layers.Dropout(dropout_rate=0.25)(inner_features)

        # We will be using softmax at the error function (cross entropy with softmax)
        with cntk.layers.default_options(init=cntk.glorot_uniform(), activation=None):
            dense_branches = []
            for branch in range(self.num_dense_branches):
                dense_branches.append(
                    cntk.layers.Dense(
                        self.num_class,
                        name='Dense {}'.format(branch + 1)
                    )(inner_features)
                )

            final_model = cntk.combine(dense_branches)

        return final_model

    def define_variables(self):
        """
        Defines the input and output variables for the neural network
        :param shape: the shape of the input variable
        :param num_class: the number of classes the output variable can have
        :return: the input and output variable
        """

        input_var = cntk.input_variable(shape=self.shape)

        output_vars = []
        for i in range(self.num_dense_branches):
            output_vars.append(
                cntk.input_variable(shape=self.num_class)
            )

        return input_var, output_vars

    def define_loss_criterion(self, model, output_vars):
        """
        Defines the loss functions and the classification error functions
        :param model: The model from which the loss function is to be defined.
        :param output_vars: a list of the output variables from the network
        :return: the loss function and the classification error function
        """

        losses = None
        classification_errors = None
        for i, output_variable in enumerate(output_vars):
            losses += cntk.cross_entropy_with_softmax(model.outputs[i], output_variable)
            classification_errors += cntk.classification_error(model.outputs[i], output_variable)

        return losses, classification_errors

    def train(self, model, num_epox=50, num_batches_per_epoch=800, batch_size=32):

        data_generator = DataGenerator(
            batch_size,
            self.num_dense_branches,
            height=self.shape[1],
            width=self.shape[2],
            num_channels=self.shape[0]
        )

        learning_rate = 1.0

        learner = cntk.adadelta(model.parameters, lr=learning_rate, rho=0.95)

        input_var, output_var = self.define_variables()

        loss, label_error = self.define_loss_criterion(model, output_var)

        trainer = cntk.Trainer(model, (loss, label_error), [learner])

        # The actual training part.
        for epoch_number in range(num_epox):
            minibatch_losses = []
            for _ in tqdm(range(num_batches_per_epoch),
                          ncols=90,
                          smoothing=1,
                          desc='Epoch {}/{}'.format((epoch_number + 1), num_epox)):

                # Generate the data.
                x, y = data_generator.generate_dataset()

                # Create the mapping between input and output data
                input_dict = {
                    input_var: x
                }

                # Each item in output var should represent one of the output from our actual data
                for i, var in enumerate(output_var):
                    input_dict[output_var[i]] = y[i]

                minibatch_losses = []
                trainer.train_minibatch(input_dict)
                minibatch_losses.append(trainer.previous_minibatch_loss_average)

            print('Average training loss after {0} epoch out of {1}: {2}'.format(
                epoch_number,
                num_epox,
                np.mean(minibatch_losses))
            )
