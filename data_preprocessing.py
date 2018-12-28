import numpy as np
import h5py
import matplotlib.pyplot as plt

##########################################################################
# TODO: Implement the data preprocessing step. Return three  dats sets.  #
# training_data, validation_data, test_data                              #
##########################################################################


'''
Data Loading
'''
with h5py.File('images_training.h5', 'r') as H:
    data = np.copy(H['data'])
with h5py.File('labels_training.h5', 'r') as H:
    label = np.copy(H['label'])
with h5py.File('images_testing.h5', 'r') as H:
    test = np.copy(H['data'])
with h5py.File('labels_testing_2000.h5', 'r') as H:
    test_label = np.copy(H['label'])


# if wanna preview the image, uncomment following code

for i in range(30):
    plt.subplot(5, 6, i + 1)
    img = plt.imshow(data[i], interpolation='none', cmap='Greys')
    plt.axis('off')

plt.show()

# combine data and label together for easy loop
def data_comb(data, label):
    # Combine data and label together.
    tuple_a = tuple(np.array([data], dtype='float32'))
    tuple_b = tuple(np.array([label], dtype='int64'))
    tuple_c = tuple(tuple_a + tuple_b)
    return tuple_c

# PCA for feature extraction, which will be tested later,
# n is the dim what you to reduce to.
def pca(data, n):
    normalized = data - data.mean(axis=0)
    covariance = np.dot(normalized.T, normalized) / len(normalized)
    # Calculation eignvector by using SVD
    u, s, v = np.linalg.svd(covariance)
    transformed_data = np.matmul(u[:, :n].T, data.T).T
    return transformed_data

"""

    data spliting format:
    - Training set: 0:28000 (in order to increase acc, will increase to 0:30000 later)
    - Validation set: 28000:30000 (will be ignore during testing the final result)
    - test set: 0:2000 which the label will be submit in h5py file.

"""
def data_category():
    # Separate the entire data to three parts.
    training_data = data_comb(data[0:30000] / 255, label[0:30000])
    validation_data = data_comb(data[28000:30000] / 255, label[28000:30000])
    test_data = data_comb(test[0:2000] / 255, test_label[0:2000])
    test_data_part2 = data_comb(test[0:5000] / 255, np.zeros([5000, ], int))
    return training_data, validation_data, test_data_part2


def label_to_vector(x):
    # convert a digit(0...9) into a corresponding vector.
    vector = np.zeros((10, 1))
    vector[x] = 1.0
    return vector


def data_load_and_preprocessing():
    # Original data will be reshaped to new dimension matrix.
    train, validation, test = data_category()

    training_inputs = [np.reshape(x, (784, 1)) for x in train[0]]
    training_label = [label_to_vector(y) for y in train[1]]
    training_data = list(zip(training_inputs, training_label))

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation[0]]
    validation_label = [label_to_vector(y) for y in validation[1]]
    validation_data = list(zip(validation_inputs, validation_label))

    test_inputs = [np.reshape(x, (784, 1)) for x in test[0]]
    test_label = [label_to_vector(y) for y in test[1]]
    test_data = list(zip(test_inputs, test_label))

    return training_data, validation_data, test_data
