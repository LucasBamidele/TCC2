from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam

# import numba as nb

HIDDEN_SIZE_ACTOR = 150
HIDDEN_SIZE_CRITIC = 256
LR = 1e-4
LR2 = 1e-4
LOSS_CLIPPING = 0.2
ENTROPY_LOSS = -1e-4
C1 = 0.5
NUM_LAYERS = 2
NUM_INPUTS = 15
NUM_OUTPUTS = 9 

# @nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction, rewards, values):
	def loss(y_true, y_pred):
		# prob = abs(y_true * y_pred)
		# old_prob = abs(y_true * old_prediction)
		# r = prob/(old_prob + 1e-10)
		newpolicy_probs = y_pred
		r = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(old_prediction + 1e-10))

		actor_loss = -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
		critic_loss = K.mean(K.square(rewards - values))
		entropy = K.mean(-(newpolicy_probs * K.log((newpolicy_probs + 1e-10))))
		# crit_loss = K.pow(y_true - y_pred, 2)
		# critic_loss = -K.mean(crit_loss)
		myloss = C1*critic_loss + actor_loss + ENTROPY_LOSS * entropy 
		return myloss
	return loss

def Actor(num_inputs, num_outputs, hidden_layer, load=''):
	state_input = Input(shape=(NUM_INPUTS,))
	advantage = Input(shape=(1,))
	rewards = Input(shape=(1,))
	values = Input(shape=(1,))
	old_prediction = Input(shape=(NUM_OUTPUTS,))

	x = Dense(HIDDEN_SIZE_ACTOR, activation='relu')(state_input)
	x = Dense(HIDDEN_SIZE_ACTOR, activation='relu')(x)
	x = Dense(HIDDEN_SIZE_ACTOR, activation='relu')(x)
	out_actions = Dense(NUM_OUTPUTS, activation='softmax', name='output')(x)

	model = Model(inputs=[state_input, advantage, old_prediction, rewards, values], outputs=[out_actions])
	model.compile(optimizer=Adam(lr=LR),
	              loss=[proximal_policy_optimization_loss(
	                  advantage=advantage,
	                  old_prediction=old_prediction,
	                  rewards=rewards,
	                  values=values)])
	# model.compile(optimizer=Adam(lr=LR),
	#               loss=['mse'])
	model.summary()
	if(load):
		model.load_weights(load)

	return model

def Critic(num_inputs, num_outputs, hidden_layer, load=''):
	state_input = Input(shape=(NUM_INPUTS,))
	x = Dense(HIDDEN_SIZE_CRITIC, activation='relu')(state_input)
	for _ in range(NUM_LAYERS - 1):
	    x = Dense(HIDDEN_SIZE_CRITIC, activation='relu')(x)

	out_value = Dense(1)(x)

	model = Model(inputs=[state_input], outputs=[out_value])
	model.compile(optimizer=Adam(lr=LR2), loss='mse')
	if(load):
		model.load_weights(load)

	return model

