import tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reproducibility
seed1=190
tensorflow.random.set_seed(seed1)
np.random.seed(seed1)
random.seed(seed1)

df = pd.read_csv('datasets.csv',header = None,names=["image_tag", "left", "top", "right","bottom",'a'])
df=df.drop(['a'], axis=1)

# normalise locations (output coordinates)
df["left"]=df["left"]/224
df["top"]=df["top"]/224
df["right"]=df["right"]/224
df["bottom"]=df["bottom"]/224

# train and test split
rt=0.2
ix=int((1-rt)*len(df))
df1 = df.iloc[:ix,:] 
df2 = df.iloc[ix+1:,:]


datagen = ImageDataGenerator(rescale=1./255)
train_g = datagen.flow_from_dataframe(
    df1, directory='./datasets/images/',
    x_col="image_tag",y_col=["left", "top", "right","bottom"],
    target_size=(224, 224),batch_size=5, 
    class_mode="raw",subset="training")
valid_g = datagen.flow_from_dataframe(
    df2, directory='./datasets/images/',
    x_col="image_tag",y_col=["left", "top", "right","bottom"],
    target_size=(224, 224),batch_size=5, 
    class_mode="raw",subset="training")


# Model Setup
model = tf.keras.models.Sequential()
model.add(tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="relu"))
model.summary()
model.compile(tf.keras.optimizers.SGD(learning_rate=0.1),loss='categorical_crossentropy',metrics=['accuracy'])
# Training
model.fit(train_g, steps_per_epoch=17, validation_data=valid_g, validation_steps=4, epochs=20)

