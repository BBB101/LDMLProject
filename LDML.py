import math
import numpy
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow import keras
from emnist import extract_test_samples, extract_training_samples
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def LoadImage(img_name):
    img = ~mpimg.imread(img_name)
    img = np.delete(img, 2, 2)
    img = np.delete(img, 1, 2)
    img = img.squeeze()
    img = clearNoise(img)
    return img


def getYStartAndEnd(img):
    start = -1
    end = -1
    for i in range(len(img)):
        row = img[i]
        if start == -1 and not isBlankWhite(row):
            start = i
        elif start != -1 and isBlankWhite(row):
            end = i
            break
    return start, end


def centerCharacter(img, img_dict, height):
    white_start, num_start, num_end, white_end = img_dict.values()
    img = img.copy()
    recommended_diff_x = int((1 / 3) * (num_end - num_start))
    diff_x = min(num_start - white_start, white_end - num_end, recommended_diff_x)
    img = img[:, num_start - diff_x:num_end + diff_x]
    y_start, y_end = getYStartAndEnd(img)
    recommended_diff_y = int((1 / 3) * (y_end - y_start))
    diff_y = min(y_start - 0, height - y_end, recommended_diff_y)
    img = img[y_start - diff_y:y_end + diff_y, :]
    resizer = Image.fromarray(img)
    resizer = resizer.resize((28, 28))
    img = np.array(resizer)
    img = clearNoise(img)
    return img


def isBlankWhite(line):
    for j in range(len(line)):
        if line[j] != 0:
            return False
    return True


def show(img):
    if img is None:
        return
    try:
        plt.imshow(img, cmap=plt.cm.binary)
    except:
        plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()


def predict(img, model):
    try:
        lst = model.predict(np.expand_dims(img, 0))
    except:
        lst = model.predict(img)

    index = 0
    val = lst[0][0]
    x = len(lst[0])
    """if x > 30:
        x = 10"""
    for i in range(1, x):
        if lst[0][i] >= 0.05:
            print(f"{i}:{round(lst[0][i], 5)}")
        if lst[0][i] > val:
            index = i
            val = lst[0][i]
    return index, val


def clearNoise(img):
    img = img.copy()
    for i in img:
        for j in range(len(i)):
            if i[j] < 130:
                i[j] = 0
    return img


def train_model():
    x = 'letters' # Can be 'digits' for training a digit model
    (trainL_img, trainL_label), (testL_img, testL_label) = extract_training_samples(x), extract_test_samples(x)
    plus = -1
    newTestLabel, newTrainLabel = [], []
    for i in range(len(trainL_label)):
        np.rot90(trainL_img[i]) # Rotates Image
        newTrainLabel.append(trainL_label[i] + plus)
    for i in range(len(testL_label)):
        np.rot90(testL_img[i])
        newTestLabel.append(testL_label[i] + plus)
    newTrainLabel = numpy.array(newTrainLabel)
    newTestLabel = numpy.array(newTestLabel)
    p1 = numpy.random.permutation(len(newTrainLabel))
    p2 = numpy.random.permutation(len(newTestLabel))
    trainL_img, newTrainLabel = trainL_img[p1], newTrainLabel[p1]
    testL_img, newTestLabel = testL_img[p2], newTestLabel[p2]
    allOLImages = np.concatenate((trainL_img, testL_img), axis=0)
    allOLLabels = np.concatenate((newTrainLabel, newTestLabel), axis=0)
    """mnist = keras.datasets.mnist
    (trainD_img, trainD_label), (testD_img, testD_label) = mnist.load_data()
    allD_img = np.concatenate((trainD_img, testD_img), axis=0)
    allD_label = np.concatenate((trainD_label, testD_label), axis=0)
    AllTestI = np.concatenate((testL_img, testD_img), axis=0)
    AllTestL = np.concatenate((newTestLabel, testD_label), axis=0)
    allOImg = np.concatenate((allOLImages, allD_img), axis=0)
    allOLabel = np.concatenate((allOLLabels, allD_label), axis=0)"""


    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)), #First Layer - Flattens the two-dimensional array into one dimensional array
        keras.layers.Dense(128, activation='relu'), # Second layer -128 nodes in layer activated using the activation function 'relu'
        keras.layers.Dense(26, activation='softmax')  #Third Layer - States the amount of possible results from 0-25 / 0-9
    ])
    model.fit(allOLImages, allOLLabels, epochs=10, batch_size=512, verbose=1, \
              validation_data=(testL_img, newTestLabel))
    model.save('my_Lmodel')

    test_loss, test_accuracy = model.evaluate(testL_img, newTestLabel, verbose=0)
    predictions = model.predict(testL_img)


def SplitImage(img):
    n = 0
    pn = 0
    TransitionState = False
    height, width = img.shape
    plt.imsave('./DebugImages/RTemo.jpg', img, cmap=plt.cm.binary)
    img = img.swapaxes(0, 1)
    plt.imsave('./DebugImages/ITenm.jpg', img, cmap=plt.cm.binary)
    lst = []
    for i in range(len(img)):
        column = img[i]
        if n == 0:
            if not isBlankWhite(column):
                # [White start, Number start, Number end, White end]
                lst.append({'white start': 0, 'num start': -1, 'num end': -1, 'white end': -1})
                n += 1
                print(i)
                lst[len(lst)-1]['num start'] = i
        elif not TransitionState and n != 0:
            if isBlankWhite(column):
                if n >= 2:
                    TransitionState = True
                    lst.append({'white start': -1, 'num start': -1, 'num end': -1, 'white end': -1})
                n += 1
                lst[len(lst)-2]['num end'] = i
                lst[len(lst)-1]['white start'] = i
        elif TransitionState and n != 0:
            if not isBlankWhite(column):
                lst[len(lst) - 2]['white end'] = i
                lst[len(lst) - 1]['num start'] = i
                TransitionState = False
    lst[0]["white start"] = 0
    if len(lst) >= 2:
        lst.remove(lst[len(lst)-1])
    lst[len(lst)-1]["white end"] = width-1
    img = img.swapaxes(0, 1)
    newlst = []
    for i in lst:
        newlst.append(centerCharacter(img, i, height))
    return newlst


def getPrediction(img_name, modName, digOrLet):
    model = keras.models.load_model(modName)
    full_img = LoadImage(img_name)
    digitsOrLetters = list(SplitImage(full_img))
    if len(digitsOrLetters) == 0:
        return None, None
    else:
        val, acc = 0, 0
        if digOrLet:
            for i in range(len(digitsOrLetters)):
                plt.imsave(f'./DebugImages/img{i}.jpg', digitsOrLetters[i], cmap=plt.cm.binary)
                digi, acci = predict(digitsOrLetters[i], model)
                print(digi, acci)
                val += digi*math.pow(10, len(digitsOrLetters)-1-i)
                acc += acci
            return f'Result: {int(val)} | Accuracy: {round((acc/len(digitsOrLetters))*100, 3)}%'
        else:
            arr = "abcdefghijklmnopqrstuvwxyz"
            val = ""
            for i in range(len(digitsOrLetters)):
                plt.imsave(f'./DebugImages/img{i}.jpg', digitsOrLetters[i], cmap=plt.cm.binary)
                lett, acci = predict(digitsOrLetters[i], model)
                print(arr[lett], acci)
                val += arr[lett]
                acc += acci
            return f'Result: "{val}" | Accuracy: {round((acc/len(digitsOrLetters))*100, 3)}%'
