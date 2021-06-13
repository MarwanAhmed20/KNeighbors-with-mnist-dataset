import numpy as np
from skimage.feature import hog
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#import matplotlib.pyplot as plt
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train)

def grid(image):
    #print("*")
    part1 = image[0:14, 0:14]
    part2 = image[14:28, 0:14]
    part3 = image[0:14, 14:28]
    part4 = image[14:28, 14:28]

    sum_xc1 = 0
    sum_xc2 = 0
    sum_xc3 = 0
    sum_xc4 = 0
    sum_yc1 = 0
    sum_yc2 = 0
    sum_yc3 = 0
    sum_yc4 = 0
    sum_pix1 = 1
    sum_pix2 = 1
    sum_pix3 = 1
    sum_pix4 = 1
    for x in range(len(part1)):
        for y in range(len(part1[x])):
            sum_xc1 += x * part1[x][y]
            sum_xc2 += x * part2[x][y]
            sum_xc3 += x * part3[x][y]
            sum_xc4 += x * part4[x][y]
            sum_yc1 += y * part1[x][y]
            sum_yc2 += y * part2[x][y]
            sum_yc3 += y * part3[x][y]
            sum_yc4 += y * part4[x][y]
            sum_pix1 += part1[x][y]
            sum_pix2 += part2[x][y]
            sum_pix3 += part3[x][y]
            sum_pix4 += part4[x][y]

    feature_vector = [sum_xc1 / sum_pix1, sum_yc1 / sum_pix1, sum_xc2 / sum_pix2, sum_yc2 / sum_pix2,
                      sum_xc3 / sum_pix3, sum_yc3 / sum_pix3, sum_xc4 / sum_pix4, sum_yc4 / sum_pix4]
    print(feature_vector)
    return feature_vector


train_images = []
for i in range(5000):
    train_images.append(grid(x_train[i]))
train_images = np.array(train_images)


test_images = []
for i in range(50):
    test_images.append(grid(x_test[i]))
test_images=np.array(test_images)



model = KNeighborsClassifier(n_neighbors=8)
model.fit(train_images,y_train[:5000])
#print(model.score(test_images,y_test[:50]))

# img=x_train[0]
# print(grid(img))
# img=x_train[0]
# img=img[14:28, 0:14]
# plt.imshow(img, cmap="Greys")
# plt.show()