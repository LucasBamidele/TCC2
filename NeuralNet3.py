from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam

import numba as nb

HIDDEN_SIZE = 64
LR = 1e-3
LOSS_CLIPPING = 0.2
ENTROPY_LOSS = 1e-4
NUM_LAYERS = 3
NUM_INPUTS = 9
NUM_OUTPUTS = 9 

@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
	def loss(y_true, y_pred):
		prob = y_true * y_pred
		old_prob = y_true * old_prediction
		r = prob/(old_prob + 1e-10)
		return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(abs(prob + 1e-10))))
	return loss

def Actor(num_inputs, num_outputs, hidden_layer, load=''):
	state_input = Input(shape=(NUM_INPUTS,))
	advantage = Input(shape=(1,))
	old_prediction = Input(shape=(NUM_OUTPUTS,))

	x = Dense(HIDDEN_SIZE, activation='relu')(state_input)
	x = Dense(HIDDEN_SIZE, activation='relu')(x)
	x = Dense(HIDDEN_SIZE, activation='relu')(x)
	out_actions = Dense(NUM_OUTPUTS, name='output')(x)

	model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
	model.compile(optimizer=Adam(lr=LR),
	              loss=[proximal_policy_optimization_loss(
	                  advantage=advantage,
	                  old_prediction=old_prediction)])
	# model.compile(optimizer=Adam(lr=LR),
	#               loss=['mse'])
	model.summary()

	return model

def Critic(num_inputs, num_outputs, hidden_layer, load=''):
	state_input = Input(shape=(NUM_INPUTS,))
	x = Dense(HIDDEN_SIZE, activation='relu')(state_input)
	for _ in range(NUM_LAYERS - 1):
	    x = Dense(HIDDEN_SIZE, activation='relu')(x)

	out_value = Dense(1)(x)

	model = Model(inputs=[state_input], outputs=[out_value])
	model.compile(optimizer=Adam(lr=LR), loss='mse')

	return model

