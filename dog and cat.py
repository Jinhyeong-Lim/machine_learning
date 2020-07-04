import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 위노그라드 알고리즘 설정 (GPU 사용시 conv 연산이 빨라짐)
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

rootPath = './datasets/cat-and-dog'

# 이미지의 사이즈는 논문에서는 (224, 224, 3)을 사용하여, 빠른 학습을 위해 사이즈를 조정
IMG_SIZE = (150, 150, 3)

imageGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True
)

trainGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'C:/Users/default.DESKTOP-6FG4SCS/Desktop/Dog&Cat/training_set'),
    target_size=IMG_SIZE[:2],
    batch_size=20
)

testGen = ImageDataGenerator(
    rescale=1./255,
).flow_from_directory(
    os.path.join(rootPath, 'C:/Users/default.DESKTOP-6FG4SCS/Desktop/Dog&Cat/test_set'),
    target_size=IMG_SIZE[:2],
    batch_size=20,
)

from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import InceptionV3

# Fc 레이어를 포함하지 않고, imagenet기반으로 학습된 가중치를 로드한 뒤 GAP 레이어를 추가
extractor = Sequential()
extractor.add(InceptionV3(include_top=False, weights='imagenet', input_shape=IMG_SIZE))
extractor.add(layers.GlobalAveragePooling2D())

extractor_output_shape = extractor.get_output_shape_at(0)[1:]

model = Sequential()
model.add(layers.InputLayer(input_shape=extractor_output_shape))
model.add(layers.Dense(2, activation='sigmoid'))

model.summary()

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['acc'],
)

extractor.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['acc'],
)

import numpy as np


def get_features(extractor, gen, cnt):
    bs = gen.batch_size
    ds = cnt
    extractor_shapes = list(extractor.get_output_shape_at(0)[1:])

    features = np.empty([0] + extractor_shapes)
    labels = np.empty((0, 2))

    for i, (trainX, trainY) in enumerate(gen):
        features = np.append(features, extractor.predict(trainX), axis=0)
        labels = np.append(labels, trainY, axis=0)
        print('batch index: {}/{}'.format(i * bs, ds), end='\r')

        if bs * i >= cnt:
            break
    print()
    return features, labels

trainX, trainY = get_features(extractor, trainGen, 3000)
testX, testY = get_features(extractor, testGen, 1000)

epochs = 32

history = model.fit(
    trainX,
    trainY,
    epochs=epochs,
    batch_size=32,
    validation_split=.1,
)

import matplotlib.pyplot as plt


def show_graph(history_dict):
    accuracy = history_dict['acc']
    val_accuracy = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(16, 1))

    plt.subplot(121)
    plt.subplots_adjust(top=2)
    plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Trainging and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=5)

    plt.subplot(122)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=5)

    plt.show()


def smooth_curve(points, factor=.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

show_graph(history.history)

model.evaluate(testX, testY)
