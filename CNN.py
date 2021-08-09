import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#Estas imagenes estan en RGB, por lo que cada canal tiene pixeles en el rago de 0-255
# por lo tanto abrán 1 pixel por canal, y como tenemos 3(RGB), entonces tendremos 3 pixeles.
# Para entender un poco mejor, abrir este link https://blog.xrds.acm.org/wp-content/uploads/2016/06/Figure1.png

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images/255.0, test_images/255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

#Creamos el modelo

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32, 32, 3))) #32 es el numero de filtros que tendremos y 3,3 la dimension de esos filtors
model.add(layers.MaxPooling2D(2,2)) #de los filtros anteriores pooleamos otra matriz de 2x2, y como en este caso usamos MaxPooling2D, tendremos el maxmio de las casillas revisadas para ver si la imagen tiene la caracteristica deseada.
model.add(layers.Conv2D(64, (3,3), activation = 'relu')) #No tenemos que darle un input shape, ya que tomará el primero
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

#Hasta el momento lo unico que hemos hecho es ver que caracteristicas tiene cada imagen introducida en nuestra red.

model.summary()

#Como se puede ver en la anterior linea de codigo, tenemos todas esas carateristicas juntas. Entonces, usamos
#una red para clasificarlas.

model.add(layers.Flatten()) #Cogemos el output anterior que es de (4,4,64) y lo volvemos de una dimension 4x4x64=1024
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10)) #El output siempre debe de ser igual al tamaño de class_names
model.summary()

#Ya que hemos separados las caracteristicas y clasificado dichas caracteristicas,
#entrenaremos el modelo.

model.compile(optimizer = 'adam',
              loss = tf.keras.losses.sparse_categorical_crossentropy(from_logits = True),
              metrics = ['accuracy'])

history = model.fit(train_images, train_labels, epochs = 10,
                    validation_data = (test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

#Tendremos una veracidad del 70% sin embargo es dificil aumentarla, ya que no tenemos suficientes imagenes para recoletar
#informarcion, las mejores CNN (Convolutional neural network) estan hechas con millones de informacion. Hay varias de estar
#redes al uso libre, y podemos usarlas como base para nuestra red.
