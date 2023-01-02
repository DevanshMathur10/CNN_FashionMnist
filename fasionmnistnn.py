import tensorflow as tf
keras=tf.keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization
from keras import Sequential


labels=['Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model=Sequential()
model.add(Conv2D(32,(5,5),activation='relu',input_shape=(28,28,1),padding='same'))
model.add(MaxPool2D((3,3)))
model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation='relu',padding='same'))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.summary()
history=model.fit(x_train,y_train,epochs=30,batch_size=30)

test_loss, test_acc = model.evaluate(x_test,  y_test)
print(f"Theaccuracy on test set is : {(test_acc*100):.6f}")

x=int(input("Choose an image between 1-9999 : "))
while x!=0:
    image=x_train[x]
    plt.imshow(image, cmap='gray')
    num=model.predict(image.reshape(1,28,28,1))
    num=np.argmax(num)
    plt.title(f"My neural network predicts {num} !")
    plt.show()
    x=int(input("Choose an image between 1-9999 : ")) 

if input("Want to save model?")=="Y":
    model.save('mnist.model3')
