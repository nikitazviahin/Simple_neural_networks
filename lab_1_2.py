import numpy as np
from prettytable import PrettyTable

train_set_input = np.array([[3]])
train_set_output = np.array([0.3]).T

EPSILON = 0.1
COLUMNS = ["Iteration", "Y", "Error"]


def save_weights(data):
    np.savetxt("weights_1_2.csv", data, delimiter=',')


class Perceptron:
    def __init__(self):
        """
            Initialize start random weights
            dimension is 4x1
        """

        # For generation the same random weights every time
        np.random.seed(1)
        # self.synaptic_weights = 2 * np.random.random((2, 1)) - 1
        self.table = PrettyTable()
        self.synaptic_weights = np.loadtxt("weights_1_2.csv", delimiter=',')

    @staticmethod
    def __sigmoid(x):
        """
        Calculate sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __sigmoid_derivative(x):
        """
        Calculate derivative of sigmoid function
        """
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, max_number_of_training_iterations):
        """
        Trainings iterations

        :param training_set_inputs: matrix of training set
        :param training_set_outputs: matrix of results of training set
        :param max_number_of_training_iterations: int, for breaking the loop if error > epsilon
        :return:
        """
        iterations = []
        outputs = []
        errors = []

        y_third_layout = 0
        error = 0
        i = 0
        print("Start weights:", str(self.synaptic_weights))
        for iteration in range(max_number_of_training_iterations):
            x_second_layout = self.synaptic_weights[0] * training_set_inputs
            y_second_layout = self.__sigmoid(x_second_layout)

            x_third_layout = self.synaptic_weights[1] * y_second_layout
            y_third_layout = self.__sigmoid(x_third_layout)

            error = abs((training_set_outputs - y_third_layout) / training_set_outputs)

            if (iteration + 1) % 10 == 0:
                iterations.append(iteration+1)
                outputs.append(round(y_third_layout[0][0], 6))
                errors.append(round(error[0][0], 6))

            if error <= EPSILON:
                print(self.synaptic_weights)
                break

            q_third_layout = self.__sigmoid_derivative(y_third_layout) * (training_set_outputs - y_third_layout)
            q_second_layout = self.__sigmoid_derivative(y_second_layout) * (q_third_layout * self.synaptic_weights[1])

            delta_weights_third_layout = q_third_layout * y_second_layout
            delta_weights_second_layout = q_second_layout * training_set_inputs

            self.synaptic_weights[0] += float(delta_weights_second_layout)
            self.synaptic_weights[1] += float(delta_weights_third_layout)
            i += 1

        save_weights(self.synaptic_weights)
        iterations.append(i + 1)
        outputs.append(round(y_third_layout[0][0], 6))
        errors.append(round(error[0][0], 6))

        self.table.add_column(COLUMNS[0], iterations)
        self.table.add_column(COLUMNS[1], outputs)
        self.table.add_column(COLUMNS[2], errors)

        print(self.table)

        print("Weights:", str(self.synaptic_weights))

    def activate(self, inputs):
        """
        Pass inputs through our neural network

        :param inputs: training example
        :return: result of neural network
        """

        x_second_layout = self.synaptic_weights[0] * inputs
        y_second_layout = self.__sigmoid(x_second_layout)

        x_third_layout = self.synaptic_weights[1] * y_second_layout
        y_third_layout = self.__sigmoid(x_third_layout)

        return y_third_layout


def main():
    perceptron = Perceptron()

    # perceptron.train(train_set_input, train_set_output, 1000)

    result = perceptron.activate(train_set_input)
    print("Режим розпізнавання: ")
    print("Початковий вектор:", str(train_set_input))
    print("Розпізнаний образ: ", str(result))


if __name__ == "__main__":
    main()
