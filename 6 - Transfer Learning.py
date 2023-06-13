import matplotlib.pyplot as plt

import tensorflow as tf
#/content/cats_and_dogs_filtered.zip

url='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip=tf.keras.utils.get_file('cats_and_dogs.zip',origin=url,extract=True)
import os
PATH=os.path.join(os.path.dirname(path_to_zip),'cats_and_dogs_filtered')
train_dir=os.path.join(PATH,'train')
validation_dir=os.path.join(PATH,'validation')
train_dataset=tf.keras.utils.image_dataset_from_directory(train_dir,shuffle=True,batch_size=32,image_size=(160,160))

validation_dataset=tf.keras.utils.image_dataset_from_directory(validation_dir,shuffle=True,batch_size=32,image_size=(160,160))

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)


print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
#print(feature_batch.shape)

base_model.trainable = False

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics='accuracy')

history = model.fit(train_dataset,
                    epochs=3,
                    validation_data=validation_dataset)

