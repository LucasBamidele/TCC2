def generate_train_from_batch2(self, batch):
	x_train = []
	angv1_train = []
	linv1_train = []
	angv2_train = []
	linv2_train = []
	angv3_train = []
	linv3_train = []
	decay = ALPHA
	mb_len = len(batch)

	old_states = np.zeros(shape=(mb_len,NUM_FEATURES))
	rewards = np.zeros(shape=(mb_len,))
	lin_actions = np.zeros(shape=(mb_len,NUMBER_OF_PLAYERS,))
	ang_actions = np.zeros(shape=(mb_len,NUMBER_OF_PLAYERS,))
	new_states = np.zeros(shape=(mb_len, NUM_FEATURES))
	
	for i, memory in enumerate(batch):
		old_state_m, action_number_ang, action_number_lin, reward_m, new_state_m = memory
		old_states[i, :] = old_state_m.transpose()[...]
		lin_actions[i] = action_number_lin
		ang_actions[i] = action_number_ang
		rewards[i] = reward_m
		new_states[i, :] = new_state_m.transpose()[...]
	old_qvals = self.model.predict(old_states, batch_size=mb_len)
	new_qvals = self.model.predict(new_states, batch_size=mb_len)
	maxQs = [[np.max(new_qval, axis=1)] for new_qval in new_qvals]
	# maxQs = np.max(new_qvals[0], axis=1)
	y = old_qvals[:]
	indexes_angs = np.linspace(0,NB_ANG_ACT, NB_ANG_ACT+1, dtype=int)
	indexes_lins = np.linspace(0,NB_LIN_ACT, NB_ANG_ACT+1, dtype=int)
	for i in range(len(y)):
		reward = rewards[i]
		y[i][,lin_actions] = reward
		exit()
	print(y)
	exit()


	y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (GAMMA * maxQs[non_term_inds])
	angv1_train = y[0]
	linv1_train = y[1]
	angv2_train = y[2]
	linv2_train = y[3]
	angv3_train = y[4]
	linv3_train = y[5]

	X_train = old_states
	y_train = y
	return X_train, angv1_train, linv1_train, angv2_train, linv2_train, angv3_train, linv3_train
