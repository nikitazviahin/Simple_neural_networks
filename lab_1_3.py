import numpy as np
from prettytable import PrettyTable

train_set_input = np.array([[1, 1], [2, 1], [1.5, 1.5], [1, 2.5]])
train_set_output = np.array([[0.02], [0.03], [0.03], [0.035]])
#
# train_set_input = np.array([[3, 2]])
# train_set_output = np.array([[0.5]])

test_set_input = np.array([[4, 1.5]])

COLUMNS = ["Iteration", "Y", "Error"]
EPSILON = 0.1
NUM_FEATURES = len(train_set_input)


class Perceptron:
    def __init__(self):
        """
            Initialize start random weights
            dimension is 4x1
        """

        # For generation the same random weights every time
        np.random.seed(1)

        self.table = PrettyTable()
        # self.weight_hidden = 2 * np.random.random((2, 3)) - 1
        # self.weight_output = 2 * np.random.random((3, 1)) - 1

        self.weight_hidden = np.loadtxt('hidden_weights_online.csv', delimiter=',')
        self.weight_output = np.loadtxt('output_weights_online.csv', delimiter=',')

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

        output_op = 0
        error = 0
        i = 0

        print("Start hidden weights:", str(self.weight_hidden))
        print("Start output weights:", str(self.weight_output))
        for iteration in range(max_number_of_training_iterations):
            input_hidden = np.dot(training_set_inputs, self.weight_hidden)

            # Output from hidden layer
            output_hidden = self.__sigmoid(input_hidden)

            # Input for output layer
            input_op = np.dot(output_hidden, self.weight_output)

            # Output from output layer
            output_op = self.__sigmoid(input_op)

            error = abs((training_set_outputs - output_op) / training_set_outputs)

            if (iteration + 1) % 10 == 0:
                iterations.append(iteration+1)
                outputs.append(round(output_op[0][0], 6))
                errors.append(round(error[0][0], 6))

            if error <= EPSILON:
                break

            q_output_layout = self.__sigmoid_derivative(output_op) * (training_set_outputs - output_op)
            delta_output_layout = q_output_layout * output_hidden

            delta_output_layout = np.reshape(delta_output_layout, (3, 1))
            q_hidden_layout_1 = self.__sigmoid_derivative(
                output_hidden[0][0]) * (q_output_layout * self.weight_output[0])
            delta_hidden_layout_1 = q_hidden_layout_1 * training_set_inputs

            q_hidden_layout_2 = self.__sigmoid_derivative(
                output_hidden[0][1]) * (q_output_layout * self.weight_output[1])
            delta_hidden_layout_2 = q_hidden_layout_2 * training_set_inputs

            q_hidden_layout_3 = self.__sigmoid_derivative(
                output_hidden[0][2]) * (q_output_layout * self.weight_output[2])
            delta_hidden_layout_3 = q_hidden_layout_3 * training_set_inputs

            delta_hidden_layout = np.array(
                [[delta_hidden_layout_1[0][0], delta_hidden_layout_2[0][0], delta_hidden_layout_3[0][0]],
                 [delta_hidden_layout_1[0][1], delta_hidden_layout_2[0][1], delta_hidden_layout_3[0][1]]])
            self.weight_output += delta_output_layout
            self.weight_hidden += delta_hidden_layout

            i += 1

        save_result(self.weight_hidden, self.weight_output)

        iterations.append(i + 1)
        outputs.append(round(output_op[0][0], 6))
        errors.append(round(error[0][0], 6))

        self.table.add_column(COLUMNS[0], iterations)
        self.table.add_column(COLUMNS[1], outputs)
        self.table.add_column(COLUMNS[2], errors)

        print(self.table)

        print("Hidden weights:", str(self.weight_hidden))
        print("Output weights:", str(self.weight_output))

    def online_train(self, training_set_inputs, training_set_outputs, epochs):
        outputs = []
        errors = []
        iterations = []

        print("Start hidden weights:", str(self.weight_hidden))
        print("Start output weights:", str(self.weight_output))
        for epoch in range(epochs):
            input_hidden = np.dot(training_set_inputs, self.weight_hidden)

            # Output from hidden layer
            output_hidden = self.__sigmoid(input_hidden)

            # Input for output layer
            input_op = np.dot(output_hidden, self.weight_output)

            # Output from output layer
            output_op = self.__sigmoid(input_op)

            outputs.append(np.reshape(output_op, (1, 4))[0])
            error = abs((training_set_outputs - np.reshape(output_op, (4, 1))) / np.reshape(output_op, (4, 1)))

            errors.append(np.reshape(error, (1, 4))[0])
            iterations.append(epoch + 1)

            for i in range(NUM_FEATURES):
                q_output_layout = self.__sigmoid_derivative(output_op[i]) * (training_set_outputs[i] - output_op[i])
                delta_output_layout = q_output_layout * output_hidden[i]
                delta_output_layout = np.reshape(delta_output_layout, (3, 1))

                q_hidden_layout_1 = self.__sigmoid_derivative(
                    output_hidden[i][0]) * (q_output_layout * self.weight_output[0])
                delta_hidden_layout_1 = q_hidden_layout_1 * training_set_inputs[i]

                q_hidden_layout_2 = self.__sigmoid_derivative(
                    output_hidden[i][1]) * (q_output_layout * self.weight_output[1])
                delta_hidden_layout_2 = q_hidden_layout_2 * training_set_inputs[i]

                q_hidden_layout_3 = self.__sigmoid_derivative(
                    output_hidden[i][2]) * (q_output_layout * self.weight_output[2])
                delta_hidden_layout_3 = q_hidden_layout_3 * training_set_inputs[i]

                delta_hidden_layout = np.array(
                    [[delta_hidden_layout_1[0], delta_hidden_layout_2[0], delta_hidden_layout_3[0]],
                     [delta_hidden_layout_1[1], delta_hidden_layout_2[1], delta_hidden_layout_3[1]]])

                self.weight_output += delta_output_layout
                self.weight_hidden += delta_hidden_layout

        self.table.add_column(COLUMNS[0], iterations)
        self.table.add_column(COLUMNS[1], outputs)
        self.table.add_column(COLUMNS[2], errors)

        print(self.table)

        save_result(self.weight_hidden, self.weight_output)

        print("Hidden weights:", str(self.weight_hidden))
        print("Output weights:", str(self.weight_output))

    def activate(self, inputs):
        """
        Pass inputs through our neural network

        :param inputs: training example
        :return: result of neural network
        """

        # Input for hidden layer
        input_hidden = np.dot(inputs, self.weight_hidden)

        # Output from hidden layer
        output_hidden = self.__sigmoid(input_hidden)

        # Input for output layer
        input_op = np.dot(output_hidden, self.weight_output)

        # Output from output layer
        output_op = self.__sigmoid(input_op)

        return output_op


def save_result(data_1, data_2):
    np.savetxt("hidden_weights_online.csv", data_1, delimiter=',')
    np.savetxt("output_weights_online.csv", data_2, delimiter=',')


def main():
    perceptron = Perceptron()

    # perceptron.train(train_set_input, train_set_output, 10000)
    # perceptron.online_train(train_set_input, train_set_output, 1000000)

    result = perceptron.activate(test_set_input)
    print("Режим розпізнавання: ")
    print("Початковий вектор:", str(test_set_input))
    print("Розпізнаний образ: ", str(result))


if __name__ == "__main__":
    main()



