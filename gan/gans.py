#GAN Implementation in Keras

import numpy as np 
from keras.datasets import mnist
from keras.layers import Dense, Reshape, Flatten, Dropout, Input
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys

class GeneticAdvNet():

	def __init__(self):

		self.imagerows = 28
		self.imagecolumn = 28
		self.channel = 1
		self.imageshape = (self.imagerows, self.imagecolumn, self.channel)
		self.dim = 100
		
		optimizer = Adam(lr = 0.0002 , beta_1 = 0.5)
		
		self.descriminator = self.descriminator()
		self.descriminator.compile(loss = 'binary_crossentropy',
		   optimizer = optimizer, 
		   metrics = ['accuracy'])

		self.generator = self.generator()

		rand_noise = Input(shape = (self.dim, ))
		generator_image = self.generator(rand_noise)
		
		self.descriminator.trainable = False
		
		image_validity = self.descriminator(generator_image)
		
		self.combine = Model(rand_noise, image_validity)
		self.combine.compile(loss = 'binary_crossentropy', optimizer = optimizer)

	def generator(self):

		model = Sequential()
		model.add(Dense(256, input_dim= self.dim))
		model.add(LeakyReLU(alpha = 0.2))
		model.add(BatchNormalization(momentum= 0.8))
		
		model.add(Dense(512))
		model.add(LeakyReLU(alpha = 0.2))
		model.add(BatchNormalization(momentum= 0.8))
		
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha = 0.2))
		model.add(BatchNormalization(momentum= 0.8))
		
		model.add(Dense(np.prod(self.imageshape), activation= 'tanh'))
		model.add(Reshape(self.imageshape))

		model.summary()

		noise = Input(shape = (self.dim, ))

		image = model(noise)

		return Model(noise,image)


	def descriminator(self):
		model = Sequential()
		model.add(Flatten(input_shape = self.imageshape))
		model.add(Dense(512))
		model.add(LeakyReLU(0.2))
		model.add(Dense(256))
		model.add(LeakyReLU(0.2))
		model.add(Dense(1, activation='sigmoid'))

		model.summary()

		image = Input(shape = self.imageshape)
		validity = model(image)
		
		return Model(image,validity)



	def train(self, epochs, batch_size = 128, sample_interval=50):

		(X_train, _), (_, _) = mnist.load_data()

		X_train = X_train / 127.5 -1
		X_train = np.expand_dims(X_train, axis= 3)

		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		for epoch in range(epochs):


			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]

			noise = np.random.normal(0, 1, (batch_size, self.dim))

			gen_imgs = self.generator.predict(noise)

			descriminator_real_loss = self.descriminator.train_on_batch(imgs, valid)
			descriminator_fake_loss = self.descriminator.train_on_batch(gen_imgs, fake)
			descriminator_loss = 0.5 * np.add(descriminator_real_loss, descriminator_fake_loss)


			noise = np.random.normal(0, 1, (batch_size, self.dim))

			generator_loss = self.combine.train_on_batch(noise, valid)

			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, descriminator_loss[0], 100*descriminator_loss[1], generator_loss))

			if epoch % sample_interval == 0:
				self.sample_images(epoch)

	def sample_images(self, epoch):
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, self.dim))
		gen_imgs = self.generator.predict(noise)

		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
			fig.savefig("%d.png" % epoch)
			plt.close()

if __name__ == '__main__':
	gan = GeneticAdvNet()
	gan.train(epochs=30000000, batch_size=32, sample_interval=200)
