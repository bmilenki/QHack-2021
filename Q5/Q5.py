#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.
    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.
    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).
            * gradient is a real NumPy array of size (5,).
            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    def parameter_shift_term(qnode, params, i, shift):
        shifted = params.copy()
        shifted[i] += shift
        forward = qnode(shifted)  # forward evaluation

        shifted[i] -= shift * 2
        backward = qnode(shifted)  # backward evaluation

        return [((1/np.sin(shift))*.5 * (forward - backward)), [forward, backward]]

    def parameter_shift(qnode, params, shift):
        gradients = np.zeros([len(params)])
        devEval = []

        for i in range(len(params)):
            ans = parameter_shift_term(qnode, params, i, shift)
            gradients[i] = ans[0]
            devEval.append(ans[1])

        return gradients, devEval

    def hessianCalc(devEval,qnode,params,shift):
        f = qnode(params)
        shiftedAnswers = np.zeros([5, 5], dtype=np.float64)
        for i in range(len(params)):
            for j in range(i+1,len(params)):
                weights = params.copy()
                weights[i] = weights[i] + shift
                weights[j] = weights[j] + shift
                s1 = qnode(weights)

                weights = params.copy()
                weights[i] = weights[i] - shift
                weights[j] = weights[j] + shift
                s2 = qnode(weights)

                weights = params.copy()
                weights[i] = weights[i] + shift
                weights[j] = weights[j] - shift
                s3 = qnode(weights)

                weights = params.copy()
                weights[i] = weights[i] - shift
                weights[j] = weights[j] - shift
                s4 = qnode(weights)

                # print(s1, s2, s3, s4)
                # print("answer:",(s1-s2-s3+s4)/((2*np.sin(shift))**2))

                shiftedAnswers[i][j] = (s1-s2-s3+s4)/((2*np.sin(shift))**2)
                shiftedAnswers[j][i] = shiftedAnswers[i][j]

        # calc diagonal using gradients/shifted answers
        for k in range(len(params)):
            forward = devEval[k][0]
            backward = devEval[k][1]

            shiftedAnswers[k][k] = (forward - 2*f + backward)/2

        #print(shiftedAnswers)
        return shiftedAnswers

    gradient, devEval = parameter_shift(circuit, weights, np.pi / 2)

    hessian = hessianCalc(devEval,circuit, weights, np.pi / 2)

    #print(gradient[0] * parameter_shift(circuit, gradient) + )

    #hessian = hessian(gradient,circuit,weights)
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )