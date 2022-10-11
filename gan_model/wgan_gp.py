import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, ReLU, TimeDistributed
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import RandomNormal
import os
import time


def Generator(input_dim, output_dim, feature_size, weight_initializer) -> tf.keras.models.Model:    
    model = tf.keras.Sequential()
    model.add(Conv1D(32, kernel_size=2, strides=1, 
                     padding='same',kernel_initializer= weight_initializer, 
                     batch_input_shape=(None,input_dim,feature_size)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Bidirectional(LSTM(64, activation='relu', kernel_initializer= weight_initializer, return_sequences=False, 
                                 dropout=0.3, recurrent_dropout=0.0)))
    model.add(Flatten())

    model.add(Dense(64, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    
    model.add(Dense(output_dim))
    return model

def Discriminator(weight_initializer, n_steps_in, n_steps_out) -> tf.keras.models.Model:
    model = tf.keras.Sequential()
    model.add(Conv1D(32, kernel_size=2, strides=1,
                     kernel_initializer= weight_initializer, padding='same', 
                     input_shape=(n_steps_in + n_steps_out, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv1D(64, kernel_size=2, strides=1,
                     kernel_initializer= weight_initializer, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())

    model.add(Dense(64, activation='linear', use_bias=True))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='linear', use_bias=True))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='linear'))
    return model

# Train WGAN-GP model
class WGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, n_steps_in, n_steps_out):
        super(WGAN, self).__init__()
        self.d_optimizer = tf.keras.optimizers.Adam(0.0004, beta_1=0.5, beta_2=0.9)
        self.g_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
        self.generator = generator
        self.discriminator = discriminator
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.batch_size = 32

    def gradient_penalty(self, batch_size, real_output, generated_output):
        """ Calculates the gradient penalty."""
        # get the interpolated data
        alpha = tf.random.normal([batch_size, self.n_steps_in + self.n_steps_out, 1], 0.0, 1.0) 
        diff = generated_output - tf.cast(real_output, tf.float32)
        interpolated = tf.cast(real_output, tf.float32) + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated data.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated data.
        grads = gp_tape.gradient(pred, [interpolated])[0]

        # 3. Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))

        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_input, real_price, past_y = data
        batch_size = tf.shape(real_input)[0]
        
        #Train the discriminator (suggested: 5 times)
        for i in range(5):
            with tf.GradientTape() as d_tape:
                # generate fake output
                generated_data = self.generator(real_input, training=True)
                # reshape the data
                generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
                generated_output = tf.concat([generated_data_reshape, tf.cast(past_y, tf.float32)], axis=1)
                real_y_reshape = tf.reshape(real_price, [real_price.shape[0], real_price.shape[1], 1])
                real_output = tf.concat([tf.cast(real_y_reshape, tf.float32), tf.cast(past_y, tf.float32)], axis=1)
                # Get the logits for the real data
                D_real = self.discriminator(real_output, training=True)
                # Get the logits for the generated data
                D_generated = self.discriminator(generated_output, training=True)
                # Calculate discriminator loss using generated and real logits
                real_loss = tf.cast(tf.reduce_mean(D_real), tf.float32)
                generated_loss = tf.cast(tf.reduce_mean(D_generated), tf.float32)
                d_cost = generated_loss - real_loss
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_output, generated_output)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * 10
            
            # Get the gradients w.r.t the discriminator loss
            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train the generator (default: 1 time)
        for i in range(1):
            with tf.GradientTape() as g_tape:
                # generate fake output
                generated_data = self.generator(real_input, training=True)
                # reshape the data
                generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
                generated_output = tf.concat([generated_data_reshape, tf.cast(past_y, tf.float32)], axis=1)
                # Get the discriminator logits for fake data
                G_generated = self.discriminator(generated_output, training=True)
                # Calculate the generator loss
                lambda_1 = 0.5 #=> adding extra losses significatively speeds up training and convergence
                lambda_2 = 0.5 #=> adding extra losses significatively speeds up training and convergence
                g_mse = np.mean(tf.keras.losses.MSE(real_y_reshape, generated_data_reshape))
                g_sign = np.mean(np.abs(np.sign(real_y_reshape) - np.sign(generated_data_reshape)))
                #print(g_mse,g_sign)
                g_loss = (1)*(-tf.reduce_mean(G_generated)) + (lambda_1)*(g_mse) + (lambda_2)*(g_sign)
            
            # Get the gradients w.r.t the generator loss
            g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return real_price, generated_data, {'discriminator_loss': d_loss, 'generator_loss': g_loss}

    def train(self, X_train, y_train, past_y, epochs):
        data = X_train, y_train, past_y
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_times'] = []
        train_hist['total_ptime'] = []


        for epoch in range(epochs):
            start = time.time()

            real_price, generated_price, loss = self.train_step(data)

            G_losses = []
            D_losses = []

            Real_price = []
            Predicted_price = []

            D_losses.append(loss['discriminator_loss'].numpy())
            G_losses.append(loss['generator_loss'].numpy())

            Predicted_price.append(generated_price)
            Real_price.append(real_price)

            # Save the model every 100 epochs
            if (epoch + 1) % 100 == 0:
                tf.keras.models.save_model(self.generator, 'gen_model_%d_%d_%d.h5' % (self.n_steps_in,self.n_steps_out,epoch))
                print('epoch', epoch+1, 'discriminator_loss', loss['discriminator_loss'].numpy(), 'generator_loss', loss['generator_loss'].numpy())

            # For printing loss
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - start
            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)
            train_hist['per_epoch_times'].append(per_epoch_ptime)
            
        # Reshape the predicted and real price
        Predicted_price = np.array(Predicted_price)
        Predicted_price = Predicted_price.reshape(Predicted_price.shape[1], Predicted_price.shape[2])
        Real_price = np.array(Real_price)
        Real_price = Real_price.reshape(Real_price.shape[1], Real_price.shape[2])

        # Plot the loss
        plt.figure(figsize=(16, 8))
        plt.plot(train_hist['D_losses'], label='D_loss')
        plt.plot(train_hist['G_losses'], label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./outcome/train_loss.png')
        #plt.show()

        return Predicted_price, Real_price