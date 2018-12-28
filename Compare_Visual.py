import h5py
import numpy as np
import matplotlib.pyplot as plt

##########################################################################
# TODO: Visualize the image and test the accuracy with 2000 correct label#
##########################################################################

with h5py.File('Predicted_labels.h5', 'r') as H:
    test_label = np.copy(H['label'])
with h5py.File('labels_testing_2000.h5', 'r') as H:
    right_label = np.copy(H['label'])
with h5py.File('images_testing.h5', 'r') as H:
    test = np.copy(H['data'])

test = test[0:2000]
test_label = test_label[0:2000]

# preview of the image before classification (cmap:hot)


def before_classify(data):
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        img = plt.imshow(data[i], interpolation='none', cmap='Greys')
        plt.axis('off')

    plt.show()

# Extract same category together.


def index_find(categ):
    index = []
    for i in range(2000):
        if test_label[i] == int(categ):
            index.append(i)
    return index


# There are 10 (0-9) categories (shoe bag etc.)
index_0 = index_find(0)
index_1 = index_find(1)
index_2 = index_find(2)
index_3 = index_find(3)
index_4 = index_find(4)
index_5 = index_find(5)
index_6 = index_find(6)
index_7 = index_find(7)
index_8 = index_find(8)
index_9 = index_find(9)
index_10 = index_find(10)

# visualize each category by using subplot (PLT)


def visual_cate(index_n):
    count1 = 1
    for i in range(500):

        if i in index_n and count1 <= 25:

            plt.subplot(5, 5, count1)
            img = plt.imshow(test[i], interpolation='none', cmap='Greys')

            plt.axis('off')
            count1 += 1

    plt.show()

# compare the result and see the accuracy.(2000 and 5000 can be adjusted)


def compare():
    count = 0
    for i, j in zip(test_label[0:2000], right_label[0:2000]):
        if i != j:
            count += 1
    print('The accuracy is: %.5f' % (1 - count / len(test_label)))


before_classify(test)
visual_cate(index_9)
compare()
