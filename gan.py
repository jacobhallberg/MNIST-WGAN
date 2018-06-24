import numpy as np
import keras
import seaborn as sns
sns.set()
from progressbar import ProgressBar
from progressbar import Bar, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Reshape, UpSampling2D, BatchNormalization, LeakyReLU, ZeroPadding2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras import optimizers
from keras.metrics import categorical_crossentropy
import matplotlib.pyplot as plt
import keras.backend as K
import random


class DCGAN():

	def __init__(self):
		self.image_rows = 28
		self.image_cols = 28
		self.channels = 1
		self.image_shape = (self.image_rows, self.image_cols, self.channels)

		# Our noise representation dimensions.
		self.latent_dimensions = 100

		self.discriminator_iterations = 5

		# Optimizer to be used for both the generator and discriminator.
		self.optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)

		# Build both the generator and discriminator.
		self.generator = self.build_generator()
		self.discriminator = self.build_discriminator()

		#########################
		#   Constructing GAN 
		#########################
		self.GAN = Sequential()
		self.GAN.add(self.generator)
		self.discriminator.trainable = False
		self.GAN.add(self.discriminator)
		self.GAN.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)


	def build_generator(self):
		
		print('Constructing Generator...')

		#########################
		#   Creating generator  
		#########################
		generator = Sequential()
		
		# Conv2DTranspose seemed to give bad results.

		# UpSampling and applying a consecutive Conv layers seems to remove 'checkerboard' generated images.
			# https://distill.pub/2016/deconv-checkerboard/
		generator.add(Dense(7*7*64, activation="relu", input_shape=(self.latent_dimensions,)))
		generator.add(Reshape((7,7,64)))

		# Upsample to increase image dimensions(Generative Portion).
		generator.add(UpSampling2D())
		# Apply convolution in hope of generalizing features.
		generator.add(Conv2D(filters=256, kernel_size=(5,5), padding="VALID"))
		# Transformation that maintains mean close to 0 and standard deviation close to 1.
			# Momentum helps converge to 'optima' faster.
		generator.add(BatchNormalization(momentum=0.8))
		generator.add(Activation("relu"))

		generator.add(UpSampling2D())
		generator.add(Conv2D(filters=128, kernel_size=(5,5), padding="VALID"))
		generator.add(BatchNormalization(momentum=0.8))
		generator.add(Activation("relu"))

		generator.add(UpSampling2D())
		generator.add(Conv2D(filters=64, kernel_size=(4,4), padding="VALID"))
		generator.add(BatchNormalization(momentum=0.8))
		generator.add(Activation("relu"))

		generator.add(Conv2D(filters=1, kernel_size=(2,2), strides=(1,1), padding="VALID", activation="tanh"))
	
		print('Generator constructed...')
		generator.summary()
					
		return generator

					
	
	def build_discriminator(self):
		
		print('Constructing Discriminator...')
					
		#########################
		# Creating Discriminator  
		#########################
		discriminator = Sequential()

		discriminator.add(Conv2D(16, kernel_size=(3,3), strides=(2,2), padding="VALID", input_shape=self.image_shape))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.20))

		discriminator.add(Conv2D(32, kernel_size=(3,3), strides=(2,2), padding="same"))
		discriminator.add(BatchNormalization(momentum=0.8))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.30))

		discriminator.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding="same"))
		discriminator.add(BatchNormalization(momentum=0.8))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.40))

		discriminator.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding="same"))
		discriminator.add(BatchNormalization(momentum=0.8))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.30))

		discriminator.add(Conv2D(256, kernel_size=(2,2), strides=(1,1), padding="same"))
		discriminator.add(BatchNormalization(momentum=0.8))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.30))

		discriminator.add(Flatten())
		# Fully connected layer             
		discriminator.add(Dense(1, activation='sigmoid'))
						
		# Sigmoid for binary classification (real = 1 or fake = 0)
		# discriminator.add(Activation('sigmoid'))

		discriminator.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)
		print('Discriminator constructed...')
		discriminator.summary()
		
		return discriminator
	
	# Using Wasserstein Loss as it shows promising results.
		# https://arxiv.org/pdf/1701.07875.pdf
	def wasserstein_loss(self, y_true, y_pred):
		'''
		 In vanilla GANs the discriminator has a sigmoid output representing the probability. However,
		 with WGAN the output is no longer constrained along the interval [0,1]. Instead the discriminator
		 is trying to make the distance (difference) between the real outputs and fake outputs as large as
		 possible. 

		 To achieve this we change the real and fake labels to 1 and -1 respectively (normally 1, 0). Thus, 
		 the loss is just the predicted output multiplied by the actual output label. 
		'''
		#y_true and y_pred are tensors and thus tensor computations are needed.
		# K.mean (keras.backend) takes mean of tensor.
		return K.mean(y_true * y_pred)

				
	def resize_image_data(self, X_train, X_test):
		# Rescales to [-1, 1], helps when updating gradients (Smaller window to move through, previously [0,255]).       
		X_train = (X_train.astype(np.float32) - 127.5) / 127.5
		# Current X_train/X_test shape is (28,28), expand makes it (28,28,1).
		X_train = np.expand_dims(X_train, axis=3)
		X_test = np.expand_dims(X_train, axis=3)
		return X_train, X_test
						
	def train(self, epochs, batch_size=100, save_interval=50): 
		# Used to create a progress bar for the epochs forloop.
		widgets = ['Test: ', Percentage(), ' ',Bar(marker='0',left='[',right=']'),' ', ETA()]
		pbar = ProgressBar(widgets=widgets, maxval=epochs+1)
		
		discriminator_losses = []
		generator_losses = []

		# Normalization:
			# Load dataset and resize images for model.
		(X_train, _), (X_test, _) = mnist.load_data()
		X_train, X_test = self.resize_image_data(X_train, X_test)

		# TODO: Possibly implement soft/noisy labels. Salimans et. al. 2016
		real_labels  =  np.ones((batch_size,1))
		fake_labels  = -np.ones((batch_size,1))
		
		pbar.start()
		for epoch in range(epochs):
			# WassersteinGAN from original paper used a 1:5 (generator:discriminator) training ratio.
			for _ in range(self.discriminator_iterations):
				#########################
				#  Train Discriminator 
				#########################

				# Grab batch of images from training data.
				real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

				# TODO: Possibly add noise to inputs, decaying overtime.
				# Sample noise from gaussian(normal) distribution to feed into generator.
				noise = np.random.normal(0, 1, (batch_size, self.latent_dimensions))   
				fake_images = self.generator.predict(noise)

				self.discriminator.trainable = True
				# Found that when training on seperate real and fake batches, performance increased.
				real_loss = self.discriminator.train_on_batch(real_images, real_labels)
				fake_loss = self.discriminator.train_on_batch(fake_images, fake_labels)
				total_loss = real_loss + fake_loss

			discriminator_losses.append(total_loss)

			#########################
			#    Train Generator 
			#########################
			self.discriminator.trainable = False
			# Train generator with random sampled noise and also real labels in an attempt to trick discriminator.
			GAN_loss = self.GAN.train_on_batch(noise, real_labels)
			generator_losses.append(GAN_loss)

			# Plot the progress
			pbar.update(epoch)
			# print ("Epoch: %d [D loss: %f] [G loss: %f]" % (epoch, total_loss, GAN_loss))

			if epoch % save_interval == 0:
				self.save_images(epoch)
		pbar.finish()
		self.plot_loss(discriminator_losses, generator_losses)

	def plot_loss(self, discriminator_losses, generator_losses):
		plt.plot(discriminator_losses, color='b', label="Discriminator Loss")
		plt.plot(generator_losses, color='r', label="Generator Loss")
		plt.legend(loc="upper right")
		plt.title("Change of Loss Over Iterations")
		plt.xlabel("Iteration")
		plt.ylabel("Loss")
		plt.show()

	def save_images(self, epoch):
		# Creating 25 images to be displayed in a 5x5 image.
		noise = np.random.normal(0, 1, (25, self.latent_dimensions))
		generated_images = self.generator.predict(noise)

		# Rescale images 0 - 1
		generated_images = 0.5 * generated_images + 1

		fig, axs = plt.subplots(5, 5)
		cnt = 0
		for i in range(5):
			for j in range(5):
				axs[i,j].imshow(generated_images[cnt, :,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("images2/mnist_%d.png" % epoch)
		plt.close()
											 
						
						
model = DCGAN()
model.train(10001)
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						
						