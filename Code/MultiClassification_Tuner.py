from keras.models import Sequential
import warnings

import cv2
import keras.callbacks
from keras.callbacks import EarlyStopping
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt


warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
import numpy as np

path_train = "data/train/"
path_test = "data/test/"
image_size = (100, 100)
batch_size = 24
epochs = 50

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.20)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

train_generator = train_datagen.flow_from_directory(
    directory=path_train,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)
valid_generator = train_datagen.flow_from_directory(
    directory=path_train,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=42
)
test_generator = test_datagen.flow_from_directory(
    directory=path_test,
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,

)

labels = {value: key for key, value in train_generator.class_indices.items()}

print("Classes present in the training and validation datasets\n")
for key, value in labels.items():
    print(f"{key} : {value}")


def build_model(hp):
    model = keras.Sequential()
    model.add(Conv2D(hp.Int('conv1_units', min_value=32, max_value=256, step=32), (3, 3), input_shape=(100, 100, 3),
                     activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(hp.Float('dropout1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Conv2D(hp.Int('conv2_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(hp.Float('dropout2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout3', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)

tuner.search(train_generator,
             validation_data=valid_generator,
             epochs=5)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
print(model.summary())

logdir = 'logs'
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
earlystop_callback = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1)

model.fit_generator(train_generator,
                    validation_data=train_generator,
                    steps_per_epoch=train_generator.n // train_generator.batch_size,
                    validation_steps=valid_generator.n // valid_generator.batch_size,
                    epochs= epochs,
                    callbacks=[tensorboard_callback,earlystop_callback])

score = model.evaluate_generator(valid_generator)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predict the classes of test images
predict = model.predict_generator(test_generator)
y_classes = np.argmax(predict, axis=1)
print(y_classes)

# # Get the labels for the classes
labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())

# Display the images along with their predicted labels'
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 12))
idx = 0

for i in range(2):
    for j in range(5):
        img_path = test_generator.filepaths[idx]
        img = cv2.imread(img_path)
        label = labels[y_classes[idx]]
        ax[i, j].set_title(f"{label}")
        ax[i, j].imshow(img)
        ax[i, j].axis("off")
        idx += 1

plt.tight_layout()
plt.suptitle("Sample Test Images", fontsize=21)
plt.show()
