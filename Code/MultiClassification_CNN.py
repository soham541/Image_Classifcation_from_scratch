
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks
import cv2
import warnings
import time

warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
import numpy as np

path_train = "data/train/"
path_test = "data/test/"
image_size = (100,100)
batch_size = 48
epochs = 50

start_time = time.time()
train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
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
    target_size= image_size,
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


def conv_model():
    model = Sequential([
        Conv2D(32, 5, padding='valid', input_shape=(100, 100, 3)),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.3),

        Conv2D(64, 3, padding='valid'),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.3),

        Conv2D(128, 3, padding='valid'),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.3),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = conv_model()
print(model.summary())

logdir = 'logs'
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
model.fit_generator(train_generator,
                    validation_data=train_generator,
                    steps_per_epoch=train_generator.n // train_generator.batch_size,
                    validation_steps=valid_generator.n // valid_generator.batch_size,
                    epochs= epochs,
                    callbacks= [tensorboard_callback])

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

end_time = time.time()
tot_time = end_time - start_time
print("Total time taken(in min ): ",tot_time/60)