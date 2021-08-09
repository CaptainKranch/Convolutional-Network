import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras


import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#Repartiremos la informacion manualmente, 80% para entrenamiento, 10% para pruebas y 10% para validacion.
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], #
    with_info=True,
    as_supervised=True,)

#Creamos una funcion para obtener los lables o etiquetas de las imagenes.

get_label_name = metadata.features['label'].int2str

# Mostramos dos imagenes de la data
for image, label in raw_train.take(5):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

#Como el tamaño de nuestras imagenes son diferentes, tenemos que convertirlas al mismo tamaño, 160x160

IMG_SIZE = 160

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

#Aplicamos la funcion .map()

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

#Testeamos como estan la imagenes

for image, label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

#mezclamos la data con .shuffle()

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

#Comparamos la imagen original con las nuevas, para ver la diferencia del shape.

for img, label in raw_train.take(2):
  print("Original shape:", img.shape)

for img, label in train.take(2):
  print("New shape:", img.shape)

#Ahora escogeremos la CNN ya hecho por tensorflow.

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.summary() #Miramos que esta CNN es muy robusta y muy bien estructurada

#Tendremos que congelar el modelo para que cunado le pongamos nuestras imagenes, no vuelva a entrenarse.

base_model.trainable = False
base_model.summary()

#Ya que lo congelamos, agreamos nuestras imagenes.

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.summary()

#Entrenamos la CNN con nuestras imagenes y la base de tensorflow congelada.

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Evaluamos para ver cual es el rendimiento antes de ponerle nuestras imagenes.
initial_epochs = 3
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

# Entrenamos con nuestras imagenes
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')


#Tambien podemos mirar  esta API de tensorflow https://github.com/tensorflow/models/tree/master/research/object_detection
