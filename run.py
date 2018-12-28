import time
from algorithm.utils import activation_function as act_fun
from algorithm.utils import parameters
from algorithm.utils import forward_and_dropout as fd
from algorithm.utils import plt_and_HDF5_result
from algorithm.utils.momentum_SGD_update import *
from algorithm.data_preprocessing import *
from algorithm.utils.weight_decay import *

"""
This file combine various ML and DL approaches, can view utils fold for detail
    -------------------------------------------------------------------------
    In the activation, I implement tanh, ReLu, softmax, cross_entropy, HingeLoss
    In the Weights , I implement weight initialization based on Xavier
    In the forward_drop, I implement dropout and forward propagation
    In the PLT_HDF5_result, I implement accuracy visualization curve
    In the data_preprocessing, I implement PCA and Data vectorization and Split
    In the weight decay, I test differert penalty coefficients
    -------------------------------------------------------------------------

There are two main parts in following: Training and Testing.

Training neural networks:
  - Forward: Similar with as NN but test different loss function (Hinge_loss).
  Based on: Yichuan Tang 2013 Deep Learning using Linear Support Vector Machines

  - BP: Using momentum SGD to increase the convergence rate
        Using mini batch to increase the stability
000000
Testing:
  - Parameters: Test different parameters
    such as learning rate and number of epoch, batch size and so on.
  - Accuracy: can reach around 86% testing on given data set.
  - Visualization: Using PLT package, view different categories (0-9)

NOTE: For large epoch, which takes longer time, the accuracy at most increase
to 95% for training data (about 1000 epoch); However, test data can only reach
at most 88% which is limited by overfitting. 30 epoches are suggested.

For efficiency, More advanced algorithms can be implemented such as BN

"""

np.seterr(all='ignore')  # To avoid 0/0 error and nan value

##########################################################################
# TODO: Load the data set and parameters for building up MLP            #
# epoch:30 batch_size=100 learning_rate=0.1 Dropout=0.9  hidden layer=3 #
##########################################################################

# Data Loading
training_data, validation_data, test_data = data_load_and_preprocessing()
# Parameters
number_of_epoch, learning_rate, batch_size, drop_out_parameter = parameters.basic_para()
# for momentum-based SGD
momentum, shift_b2, shift_b3, shift_b4, shift_b5, shift_W2, shift_W3, shift_W4, shift_W5 = parameters.momentum_SGD_para()
# Network Architecture
n_node_input, n_node_hidden, n_node_hidden2, n_node_hidden3, n_node_output, W2, W3, W4, W5, b2, b3, b4, b5 = parameters.network_para()

# Training
test_errors = []
training_errors = []
training_cross_entropys = []
training_Hinge_loss = []
test_result = []
test_accuracy = []
final_result = []

n = len(training_data)

start = time.clock()
for j in range(number_of_epoch + 1):

    # Stochastic Gradient Descent
    np.random.shuffle(training_data)

    # Mini-batch training
    sum_train_error = 0
    cross_entropy_loss = 0
    Hinge_loss = 0
    start = time.clock()
    for k in range(0, n, batch_size):
        batch = training_data[k:k + batch_size]

        # initialization gradient for samples in each batch
        batch_gradient_b5, batch_gradient_b4, batch_gradient_b3, batch_gradient_b2 = 0, 0, 0, 0
        batch_gradient_W5, batch_gradient_W4, batch_gradient_W3, batch_gradient_W2 = 0, 0, 0, 0

        # Each sample training
        for x, y in batch:
            """
            Training Part

                Each batch contain 100 elements.
                Three hidden layer will be built.
                Tanh activation and dropout will be applied to each layer
                Cross_entropy loss and Hinge loss will be compared

            """

            layer_1 = x

            z2 = fd.forward(layer_1, W2, b2)
            layer_2 = act_fun.tanh(z2)
            # Forward with dropout to avoid over fitting
            fd.drop_out(layer_2, drop_out_parameter)
            z3 = fd.forward(layer_2, W3, b3)
            layer_3 = act_fun.tanh(z3)
            # Dropout
            fd.drop_out(layer_3, drop_out_parameter)
            z4 = fd.forward(layer_3, W4, b4)
            layer_4 = act_fun.tanh(z4)

            # Dropout
            fd.drop_out(layer_4, drop_out_parameter)
            z5 = fd.forward(layer_4, W5, b5)
            prediction = act_fun.tanh(z5)

            # Cross-Entropy Cost of prediction and label

            delta_output = prediction - y

            """
            Back_propagation

                gradient descent will follow the derivative direction
                Weight decay will be applied here
                Momnentum SGD which helps keep the gradient descent in right direction

            """
            # Calculate weight and bias for hidden layer
            delta_hidden_layer4 = act_fun.tanh_derivative(
                z4) * np.dot(W5.transpose(), delta_output)
            delta_hidden_layer3 = act_fun.tanh_derivative(
                z3) * np.dot(W4.transpose(), delta_hidden_layer4)
            delta_hidden_layer2 = act_fun.tanh_derivative(
                z2) * np.dot(W3.transpose(), delta_hidden_layer3)
            # Gradient of C in terms of bias
            gradient_b5 = weight_decay(
                delta_output, learning_rate, weight_decay_rate=0.05)
            gradient_b4 = weight_decay(
                delta_hidden_layer4,
                learning_rate,
                weight_decay_rate=0.05)
            gradient_b3 = weight_decay(
                delta_hidden_layer3,
                learning_rate,
                weight_decay_rate=0.05)
            gradient_b2 = weight_decay(
                delta_hidden_layer2,
                learning_rate,
                weight_decay_rate=0.05)
            # Gradient of C in terms of weight

            gradient_W5 = weight_decay(
                np.dot(
                    delta_output,
                    layer_4.transpose()),
                learning_rate,
                weight_decay_rate=0.05)
            gradient_W4 = weight_decay(
                np.dot(
                    delta_hidden_layer4,
                    layer_3.transpose()),
                learning_rate,
                weight_decay_rate=0.05)
            gradient_W3 = weight_decay(
                np.dot(
                    delta_hidden_layer3,
                    layer_2.transpose()),
                learning_rate,
                weight_decay_rate=0.05)
            gradient_W2 = weight_decay(
                np.dot(
                    delta_hidden_layer2,
                    layer_1.transpose()),
                learning_rate,
                weight_decay_rate=0.05)

            # update gradients

            batch_gradient_b5 += gradient_b5
            batch_gradient_b4 += gradient_b4
            batch_gradient_b3 += gradient_b3
            batch_gradient_b2 += gradient_b2

            batch_gradient_W5 += gradient_W5
            batch_gradient_W4 += gradient_W4
            batch_gradient_W3 += gradient_W3
            batch_gradient_W2 += gradient_W2

            # Training Error

            sum_train_error += int(np.argmax(prediction) != np.argmax(y))
            cross_entropy_loss += np.sum(abs(delta_output))
            Hinge_loss += act_fun.HingeLoss(prediction, y)
        # update weights & biases via momentum-based SGD

        b5 += momentum_update(shift_b5, batch_gradient_b5,
                              momentum, learning_rate, batch_size)
        b4 += momentum_update(shift_b4, batch_gradient_b4,
                              momentum, learning_rate, batch_size)
        b3 += momentum_update(shift_b3, batch_gradient_b3,
                              momentum, learning_rate, batch_size)
        b2 += momentum_update(shift_b2, batch_gradient_b2,
                              momentum, learning_rate, batch_size)

        W5 += momentum_update(shift_W5, batch_gradient_W5,
                              momentum, learning_rate, batch_size)
        W4 += momentum_update(shift_W4, batch_gradient_W4,
                              momentum, learning_rate, batch_size)
        W3 += momentum_update(shift_W3, batch_gradient_W3,
                              momentum, learning_rate, batch_size)
        W2 += momentum_update(shift_W2, batch_gradient_W2,
                              momentum, learning_rate, batch_size)

    # Report Training Error
    print('**************** Start Training Epoch: %d ****************' % j)
    print("Train error: %d / %d || Accuracy is %4f" %
          (sum_train_error, n, (1 - sum_train_error / n)))
    training_errors.append(np.float(sum_train_error) / n)

    print("Cross entropy loss : " + str(cross_entropy_loss / n))
    training_cross_entropys.append(cross_entropy_loss / n)

    print("Hinge loss : " + str(np.sum(Hinge_loss) / n))
    training_Hinge_loss.append(np.sum(Hinge_loss) / n)

    """
    Test Part

        Similar with training process, however:
        Before the end of iteration, Validation  set wil be used.
        At the end of testing, 2000 elements test set will be used

        output the h5py file for submission

    """
    if j == number_of_epoch:

        for x, y in test_data:
            # At the end of iteration
            # Feed forward
            layer_1 = x
            z2 = fd.forward(layer_1, W2, b2)
            layer_2 = act_fun.tanh(z2)

            z3 = fd.forward(layer_2, W3, b3)
            layer_3 = act_fun.tanh(z3)

            z4 = fd.forward(layer_3, W4, b4)
            layer_4 = act_fun.tanh(z4)

            z5 = fd.forward(layer_4, W5, b5)

            prediction = act_fun.tanh(z5)
            final_result.append(np.argmax(prediction))

    else:
        n_test = len(validation_data)
        # before the iteration, we use validation data for testing.
        test_error = 0
        for x, y in validation_data:
            # Feed forward
            layer_1 = x
            z2 = fd.forward(layer_1, W2, b2)
            layer_2 = act_fun.tanh(z2)

            z3 = fd.forward(layer_2, W3, b3)
            layer_3 = act_fun.tanh(z3)

            z4 = fd.forward(layer_3, W4, b4)
            layer_4 = act_fun.tanh(z4)

            z5 = fd.forward(layer_4, W5, b5)

            prediction = act_fun.tanh(z5)

            # Test Error
            test_error += int(np.argmax(prediction) != np.argmax(y))

        # Report Test Error

        print("Test error  :   %d / %d || Accuracy is %4f" %
              (test_error, n_test, (1 - test_error / n_test)))
        test_errors.append(np.float(test_error) / n_test)
        test_accuracy.append(1 - test_error / n_test)
        print('**************** End Training Epoch: %d ****************\n' % j)
# Calculate the running time
end = time.clock()
print('Time used : ' + str(end - start))
# Plot and output results

plt_and_HDF5_result.plt_error_correction(
    training_errors[0:number_of_epoch], number_of_epoch)
plt_and_HDF5_result.plt_accuracy(
    test_accuracy[0:number_of_epoch], number_of_epoch)
plt_and_HDF5_result.output_result(final_result)

##########################################################################
#                             END OF YOUR CODE                           #
##########################################################################
