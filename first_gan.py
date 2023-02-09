# step 0: import modules.
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


# step 1: importing (unlabeled) data.
train_data_length = 2048
train_data = np.zeros((train_data_length, 2))
train_data[:, 0] = 2 * math.pi * np.random.rand(train_data_length)
train_data[:, 1] = np.sin(train_data[:, 0])


# # step 2: scale data.
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)


# step 2.5: visualize data.
plt.plot(train_data[:, 0], train_data[:, 1], ".")
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
#plt.plot(scaled_train_data[:, 0], scaled_train_data[:, 1], ".")
plt.title(f'Original distribution (sin curve over 0-2pi).')
plt.savefig(f"original-distribution.png", dpi=150)
plt.close()



# step 3: build the generator and discriminator models.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout

def build_generator():
    model = Sequential()
    model.add(Dense(16, input_dim=2))
    model.add(LeakyReLU(0.2))
    model.add(Dense(32))
    model.add(LeakyReLU(0.2))
    model.add(Dense(2))
    return model


def build_discriminator():
    model = Sequential()
    model.add(Dense(256, input_dim=2))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model


generator = build_generator()
#g.summary()

discriminator = build_discriminator()
#d.summary()


# step 4: setting up training. (We'll redefine the fit method for making a custom training loop for GANs.)

# ps: There must be some balance between the learning of the generator and discriminator. One should not
# overperform the other, they must progress equally.

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.utils import shuffle
from tensorflow.keras.models import Model


class FirstGAN(Model): 
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator 
        self.discriminator = discriminator 
        
    def compile(self, g_opt, d_opt, g_loss, d_loss, batch_size, *args, **kwargs): 
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss 
        self.batch_size = batch_size


    def train_step(self, batch): #called inside fit method.

        real_data = batch
        latent_space = tf.random.normal((self.batch_size, 2))
        fake_data = self.generator(latent_space, training=False)
        
        # Train the discriminator
        with tf.GradientTape() as d_tape: 

            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_data, training=True) 
            yhat_fake = self.discriminator(fake_data, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
            
            # Create labels for real and fakes images
            y_realfake = tf.concat([tf.ones_like(yhat_real), tf.zeros_like(yhat_fake)], axis=0)
            
            # Add some noise to the TRUE outputs
            # noise_real = -0.15*tf.random.uniform(tf.shape(yhat_real.shape))
            # noise_fake = 0.15*tf.random.uniform(tf.shape(yhat_fake.shape))
            # y_realfake += tf.concat([noise_real, noise_fake], axis=0)
            
            # Calculate loss - BINARYCROSS 
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)
            
        # Apply backpropagation - nn learn 
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables) 
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
        
        # Train the generator 
        with tf.GradientTape() as g_tape: 
            # Generate some new data
            latent_space = tf.random.normal((self.batch_size, 2))
            gen_data = self.generator(latent_space, training=True)
                                        
            # Create the predicted labels
            predicted_labels = self.discriminator(gen_data, training=False)
                                        
            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.ones_like(predicted_labels), predicted_labels) 
            
        # Apply backprop
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))
        
        return {"d_loss":total_d_loss, "g_loss":total_g_loss}


g_opt = Adam(learning_rate=0.001)
d_opt = Adam(learning_rate=0.001)
# g_opt = Adam(learning_rate=0.0001)
# d_opt = Adam(learning_rate=0.00001) #generator is gonna learning faster than discriminator cause its task is harder
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()
batch_size = 32
epochs = 1000

gan = FirstGAN(generator, discriminator)
gan.compile(g_opt, d_opt, g_loss, d_loss, batch_size)

import os
from tensorflow.keras.callbacks import Callback

class GanMonitor(Callback):
    def __init__(self, latent_space, scaler):
        self.latent_space = latent_space
        self.scaler = scaler


    def on_epoch_end(self, epoch, logs=None):
        generated_data = self.model.generator(self.latent_space)
        #generated_data = scaler.inverse_transform(generated_data) #scaling data to original range
        plt.plot(generated_data[:, 0], generated_data[:, 1], ".")
        plt.title(f'Generated data after epoch {epoch}.')
        plt.savefig(f"./generated-data/epoch-{epoch}.png", dpi=150)
        plt.close()

# step 5: training
hist = gan.fit(train_data, batch_size=batch_size, epochs=epochs, callbacks=[GanMonitor(np.random.randn(2048, 2), scaler)])
#hist = gan.fit(scaled_train_data, batch_size=batch_size, epochs=epochs, callbacks=[GanMonitor(np.random.randn(2048, 2), scaler)])
plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.savefig(f"./loss.png", dpi=150)
plt.close()

# step 6: save

generator.save('./models/generator.h5')
discriminator.save('./models/discriminator.h5')
