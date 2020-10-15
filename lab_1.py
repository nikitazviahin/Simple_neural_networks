import numpy as np
from numpy import exp, array, random, dot, savetxt, loadtxt
from prettytable import PrettyTable

e = 0.00001
cols = ["i", "y", "err"]

def save_weights(data):
    savetxt("weightslab11.csv", data, delimiter=',')

class NeuralNetwork:
    def __init__(this):

        # For generation the same random weights every time
        random.seed(1)
        this.syn_w = 2*random.random((4, 1)) - 1
        this.syn_w = loadtxt("weightslab11.csv", delimiter=',')
        this.table = PrettyTable()

    @staticmethod
    def sigm(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def dfSigm(x):
        return x * (1 - x)

    def train(this, tr_inputs, tr_outputs, tr_iter_limit):
        iterations = []
        outputs = []
        errors = []

        output = 0
        error = 0
        i = 0

        print("Initial weights:", str(this.syn_w))

        for iteration in range(tr_iter_limit):

            output = this.activate(tr_inputs)
            error = abs((tr_outputs - output) / tr_outputs)

            if (iteration + 1) % 10 == 0:
                iterations.append(iteration+1)
                outputs.append(np.round(output[0], 6))
                errors.append(np.round(error[0], 6))

            with open("lab_1_1_temp.txt", "a") as f:
                 f.write(str(iteration)+'\t'+str(error[0][0])+'\n')

            if error < e:
                break

            delta = this.dfSigm(output) * (tr_outputs - output)
            adjustment = dot(tr_inputs.T, delta)


            for i in range(len(this.syn_w)):
                this.syn_w[i] += adjustment[i][0]
            i += 1

        save_weights(this.syn_w)
        iterations.append(i + 1)
        outputs.append(np.round(output[0], 6))
        errors.append(np.round(error[0], 6))

        this.table.add_column(cols[0], iterations)
        this.table.add_column(cols[1], outputs)
        this.table.add_column(cols[2], errors)

        print(this.table)

        print("Weights:", str(this.syn_w))

    def activate(this, inputs):

        return this.sigm(dot(inputs, this.syn_w))


def save_result(input_data, output_data):
    with open("lab_1_1_result.txt", 'w') as f:
        f.write(str(input_data)+'\t'+str(output_data))


def main():

    neural_network = NeuralNetwork()


    #tr_inputs = array([[4, 5, 12, 3]])
    #tr_outputs = array([[0.3]]).T

    #training regime
    #print("Training regime")
    #neural_network.train(tr_inputs, tr_outputs, 1000)

    #recognition regime
    test_set = array([2, 3, 4, 5])
    test_set_output = neural_network.activate(test_set)
    print("Recognition: ")
    print("Initial vector:", str(test_set))
    print("Recognized form: ", str(test_set_output))
    print("Changed weights: ", neural_network.syn_w)


if __name__ == '__main__':
    main()
