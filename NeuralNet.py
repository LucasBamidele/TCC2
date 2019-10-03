from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback
from keras.layers import Conv1D, MaxPooling1D, Input
from time import sleep


def cnn_model(num_players, time_steps,load=''):
	my_shape = ((2 + 3*num_players))
	model = Sequential()

	model.add(Conv1D(filters=64, kernel_size=1, padding='same',
	                 input_shape=(time_steps, my_shape)
	                 ))
	model.add(Activation('relu'))
	
	model.add(Conv1D(filters=64, kernel_size=1))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(MaxPooling1D(pool_size=1))
	model.add(Dropout(0.25))

	model.add(Conv1D(filters=64, kernel_size=1))
	model.add(Activation('relu'))


	model.add(Flatten())
	
	model.add(Dense(512))
	model.add(Activation('relu'))
	


	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))


	myn = 3*4
	model.add(Dense(myn))
	model.add(Activation('softmax'))



	rms = RMSprop()
	model.compile(loss='categorical_crossentropy', optimizer=rms)
	if(load):
		model.load_weights(load)
	print(model.summary())
	return model


def neural_net_model(num_players, load=''):
	model = Sequential()
	#layer 1
	my_shape = ((5 + 7*num_players))
	# my_shape = 5
	model.add(Dense(
		512,input_shape=(my_shape,)
		))

	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))


	myn = 3*5
	model.add(Dense(myn))
	model.add(Activation('softmax'))
	

	rms = RMSprop()
	#model.compile(loss='categorical_crossentropy', optimizer=rms)
	model.compile(loss='mse', optimizer=rms)
	if(load):
		model.load_weights(load)

	return model

def neural_net_model2(num_players, load=''):
	model = Sequential()
	#layer 1
	my_shape = ((5 + 8*num_players))
	# my_shape = 5
	model.add(Dense(
		64,input_shape=(my_shape,)
		))
	model.add(Dense(128))	#128
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(128))	#128
	model.add(Activation('relu'))
	model.add(Dropout(0.2))


	myn = 11*5
	model.add(Dense(myn))
	#model.add(Activation('softmax'))
	

	rms = RMSprop()
	#model.compile(loss='categorical_crossentropy', optimizer=rms)
	model.compile(loss='mse', optimizer=rms)
	if(load):
		model.load_weights(load)

	return model

def neural_net_model2_5(num_players, load=''):
	my_shape = ((1 + 3*num_players))
	main_input = Input(shape=(my_shape,), name='main_input')

	x = Dense(128, activation='relu')(main_input)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.1)(x)

	lin_v = Dense(11, name='linear_velocity')(x)
	ang_v = Dense(5, name='angular_velocity')(x)

	model = Model(main_input, outputs=[ang_v, lin_v])
	rms = RMSprop()
	model.compile(loss='mse', optimizer=rms)
	if(load):
		print('loading model')
		model.load_weights(load)

	return model

def neural_net_model3(num_players, load=''):
	model = Sequential()
	#layer 1
	my_shape = ((5 + 8*num_players))
	# my_shape = 5
	steps = 30*4
	model.add(LSTM(
		128,input_shape=(steps,my_shape)
		))

	"""
	old model
	"""

	# model.add(Dense(256))	#128
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))

	# model.add(Dense(256))	#128
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))

	# model.add(Dense(256))	#128
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))

	"""
	new big model

	"""

	model.add(Dense(1024))	#128
	model.add(Activation('relu'))
	model.add(Dropout(0.3))

	model.add(Dense(1024))	#128
	model.add(Activation('relu'))
	model.add(Dropout(0.3))

	model.add(Dense(512))	#128
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(512))	#128
	model.add(Activation('relu'))
	model.add(Dropout(0.2))


	myn = 11*5
	model.add(Dense(myn))
	print(model.summary())
	rms = RMSprop()
	#model.compile(loss='categorical_crossentropy', optimizer=rms)
	model.compile(loss='mse', optimizer=rms)
	if(load):
		model.load_weights(load)

	return model