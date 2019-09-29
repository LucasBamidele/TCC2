"""
TODO:
	Create temporal series for states (define size)
		maybe try to predict, need to work this better
	expand to 3 vs 3 players (invert input before)
	try centralized approach first
	work on a better model for cnn and basic ann
	
"""
import numpy as np
import NeuralNet as nn
import random
from keras.utils import to_categorical
from collections import deque
from time import sleep
import sys

#TODO aceleracao em vez de velocidades
#trablahar nos hiperparametros
#aumentar recompensas vs recompensas negativas

"""
ideia principal

serie temporal de 10 frames?
manter memoria dos ultimos 10 estados e recompensas
montar batch dos proximos 10 para treino

"""
load_model = False
only_play = False
if(len(sys.argv)>1):
    if(sys.argv[1]=='play'):
        only_play = True
    if(sys.argv[1]=='load' or sys.argv[1]=='load_no_window'):
    	load_model = True


GOAL_REWARD = 500
PASS_REWARD = 100
RETAKE_REWARD = 100
ENEMY_GOAL_REWARD = -500
GAMMA = 0.95
MAX_FRAMES = 1000000
ALPHA = 0.8
EPSILON = 1	#change for 0.1

TIME_STEPS = 10

MAX_LIN_SPEED = 50
MIN_LIN_SPEED = 0
LIN_STEP = 5
MAX_ANG_SPEED = 6
MIN_ANG_SPEED = -6
ANG_STEP = 1

MAX_LIN_ACCEL = 2
MAX_ANG_ACCEL = 1

MAX_MEMORY_SPEED = 20

#NUMBER_OF_ACTIONS = ((MAX_LIN_SPEED - MIN_LIN_SPEED + LIN_STEP)//LIN_STEP)*((MAX_ANG_SPEED -MIN_ANG_SPEED+ANG_STEP)//ANG_STEP)
NUMBER_OF_ACTIONS = (MAX_ANG_ACCEL*2 +1)*(MAX_LIN_ACCEL*2+1)

NUMBER_OF_PLAYERS = 1


BALL_MAX_X = 76
BALL_MIN_X = -76





OBSERVE_TIMES = 10 #150 # BUFFER
MAX_MEMORY_BALL = 15


BATCH_SIZE = 10#150 #1000


model_name = 'saved_models/mymodel_10.h5'
MIN_DELTA_NO_MOVEMENT = 0.5
#transform an input of robot_allies, robot_opponents, and ball to a valid array
def transform_to_state(robot_allies, robot_opponents, ball):
	state = []
	state.append(ball.body.position[0])
	state.append(ball.body.position[1])
	for a in range(NUMBER_OF_PLAYERS):
		state.append(robot_allies[a].body.position[0])
		state.append(robot_allies[a].body.position[1])
		state.append(robot_allies[a].body.angle)

	state = np.array(state)
	state = state.reshape(2 + 3*NUMBER_OF_PLAYERS,1)
	#exit()
	# for ally in robot_allies:
	# 	state.append((ally.body.position[0], ally.body.position[1]))
	# for oppon in robot_opponents:
	# 	state.append((oppon.body.position[0], oppon.body.position[1]))
	# state = np.array(state)
	return state

class SimController(object):
	"""docstring for SimController"""
	def __init__(self):
		self.decrease = 0.0
		self.restart = False
		self.times = 0
		self.times_since_restart = 0
		self.iterations = 0
		self.replay_memory = deque()
		self.old_state = None
		self.action_space = []
		self.action_number = None
		for angle in range(-MAX_ANG_ACCEL, MAX_ANG_ACCEL+ 1):
			for linear in range(-MAX_LIN_ACCEL, MAX_LIN_ACCEL +1):
				self.action_space.append((angle, linear))
		self.speed = [0,0]
		self.play_mem = deque()
		self.action = None
		if(only_play or load_model):
			self.model = nn.cnn_model(NUMBER_OF_PLAYERS,TIME_STEPS, model_name)
		else :
			self.model = nn.cnn_model(NUMBER_OF_PLAYERS, TIME_STEPS)
		self.reward = {
			'goal': GOAL_REWARD,
			'pass': PASS_REWARD,
			'retake': RETAKE_REWARD,
			'enemy_goal': ENEMY_GOAL_REWARD,
		}
		self.last_speeds = deque()
		self.ball_memory = deque()
		self.player_memory = deque()
		self.t_hits = 0
		super(SimController, self).__init__()

	def add_speed_memory(self, speed):
		if(len(self.last_speeds) >= MAX_MEMORY_SPEED):
			self.last_speeds.popleft()
		self.last_speeds.append(speed[:])

	def add_ball_memory(self,ball):
		if(len(self.ball_memory) >= MAX_MEMORY_BALL):
			self.ball_memory.popleft()
		self.ball_memory.append((ball.body.position[0], ball.body.position[1]))

	def add_player_memory(self, player):
		if(len(self.player_memory) >= MAX_MEMORY_BALL):
			self.player_memory.popleft()
		self.player_memory.append((player.body.position[0], player.body.position[1]))
	#TODO
	def getReward(self, new_state, robot_allies, robot_opponents, ball):
		reward = - 1 - self.t_hits
		if(new_state[0] <= BALL_MIN_X):
			reward = self.reward['enemy_goal']
			self.restart = True
		elif(new_state[0] >= BALL_MAX_X):
			reward = self.reward['goal']
			self.restart = True
			print('goall!!!!')

		elif(self.isPlayerStuck()):
			reward = -100
			self.restart = True
			self.player_memory = deque()
		elif(self.playerHitBall(robot_allies, robot_opponents, ball)):
			self.t_hits = 0
			reward = 200
		elif(self.isSpinning()):
			reward = -100
			self.last_speeds = deque()
		self.t_hits+=1
		if(not self.isBallMoving()):
			self.ball_memory = deque()
		#evaluate the reward from the game state!
		return reward

	def isSpinning(self):
		if(len(self.last_speeds) < MAX_MEMORY_SPEED):
			return False
		return all(x == self.last_speeds[0] for x in self.last_speeds) 


	def isBallMoving(self):
		if(len(self.ball_memory) < MAX_MEMORY_BALL):
			return True
		# x = list(map(lambda x : x[0], self.ball_memory))
		# y = list(map(lambda x : x[1], self.ball_memory))
		dx = abs(self.ball_memory[0][0] - self.ball_memory[MAX_MEMORY_BALL-1][0])
		dy = abs(self.ball_memory[0][1] - self.ball_memory[MAX_MEMORY_BALL-1][1])
		if(dx < 0.1 and dy < 0.1):
			return True
		return False

		#x.pop(left)


	def isPlayerStuck(self):
		if(len(self.player_memory) < MAX_MEMORY_BALL):
			return False
		# x = list(map(lambda x : x[0], self.player_memory))
		# y = list(map(lambda x : x[1], self.player_memory))
		dx = abs(self.player_memory[0][0] - self.player_memory[MAX_MEMORY_BALL-1][0])
		dy = abs(self.player_memory[0][1] - self.player_memory[MAX_MEMORY_BALL-1][1])
		if(dx < 0.1 and dy < 0.1):
			return True
		return False

	def playerHitBall(self, robot_allies, robot_opponents, ball):
		if(robot_allies[0].body.userData == ball.body):
			print('hit!')
			robot_allies[0].body.userData = None
			return True
		return False




	def compute(self, robot_allies, robot_opponents, ball):
		new_state = transform_to_state(robot_allies, robot_opponents, ball)
		reward = self.getReward(new_state, robot_allies, robot_opponents, ball)
		if(len(self.replay_memory) < OBSERVE_TIMES):
			self.replay_memory.append((self.old_state, self.action, self.action_number, reward, new_state))
		else :
			batch = self.replay_memory
			self.replay_memory = []
			x_train, y_train = self.generate_train_from_batch(batch)
			#x_train = np.expand_dims(x_train, axis=2)
			print('shape',x_train.shape)
			x_train = x_train.reshape(TIME_STEPS, (BATCH_SIZE//TIME_STEPS), 1)
			print(x_train.shape)
			self.model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, verbose=0)

			self.replay_memory.popleft()
			self.replay_memory.append((self.old_state, self.action, self.action_number, reward, new_state))
		self.times+=1
		self.add_ball_memory(ball)
		self.add_player_memory(robot_allies[0])
		if(self.times%1500 == 0):
			self.model.save_weights(model_name) 
		if(self.times > MAX_FRAMES):
			self.model.save_weights(model_name)
			exit()

	def isTerminalState(self,reward):
		if(reward == self.reward['goal']):
			return True
		if(self.restart):
			return True
		return False


	def generate_train_from_batch(self, batch):
		x_train = []
		y_train = []
		for memory in batch:
			old_state, action, action_number, reward, new_state = memory
			old_qval = self.model.predict(old_state.transpose())
			new_qval = self.model.predict(new_state.transpose())

			# old_qval = self.model.predict(np.expand_dims(old_state.transpose(),axis=2))
			# new_qval = self.model.predict(np.expand_dims(new_state.transpose(),axis=2))
			max_qval = np.max(new_qval)	#maybe reevaluate

			y = [((1 - ALPHA)*old_qval_i + ALPHA*(reward + GAMMA*max_qval)) for old_qval_i in old_qval]
			y = np.array(y)

			# if(not self.isTerminalState(reward)):
			# 	update = reward + GAMMA*max_qval
			# 	# new_qval = (1 - ALPHA)*old_qval + ALPHA*(reward + GAMMA*max_qval)
			# else :
			# 	update = reward
			# 	# new_qval = (1 - ALPHA)*old_qval + ALPHA*reward
			# y = np.zeros((1, NUMBER_OF_ACTIONS))
			# y[:] = old_qval[:]
			# y[0][action_number]=update
			x_train.append(old_state.reshape((2 + 3*NUMBER_OF_PLAYERS),))
			y_train.append(y.reshape(NUMBER_OF_ACTIONS,))
		x_train = np.array(x_train)
		y_train = np.array(y_train)
		return x_train, y_train


	def action_saturate(self, act):
		angle_v, lin_v = act
		lin_v = min(max(MIN_LIN_SPEED, lin_v), MAX_LIN_SPEED)
		angle_v = min(max(MIN_ANG_SPEED, angle_v), MAX_ANG_SPEED)
		return [angle_v, lin_v]


	def learn_playing_descentrallized(self):
		pass

	def sync_control_centrallized(self, ally_positions, enemy_positions, ball):
		state = transform_to_state(ally_positions, enemy_positions, ball)
		self.old_state = state
		if(len(self.play_mem) >= 10)
			self.play_mem.popleft()
		self.play_mem.append(state.transpose())
		#print('state',state.transpose())
		dec = max(EPSILON - self.decrease, 0.1)
		print('EPSILON', dec)
		if((random.random() < dec or self.times < OBSERVE_TIMES) and not only_play):
			action = (random.randint(0,NUMBER_OF_ACTIONS-1))
		else :
			#predicted_qval = self.model.predict(state.transpose(), batch_size=1) #checar batch size!!
			predicted_qval = self.model.predict(np.expand_dims(self.play_mem,axis=2), batch_size=1) #checar batch size!!
			action = np.argmax(predicted_qval)
		#print(action)
		self.action_number = action
		self.action = self.action_space[action]
		self.speed[0] += self.action[0]
		self.speed[1] += self.action[1]
		self.speed = self.action_saturate(self.speed)
		self.add_speed_memory(self.speed)
		(a,b) = self.speed
		print(self.speed)
		allies = [(a,b),(0,0),(0,0),(0,0),(0,0)]
		enemies = [(0,0),(0,0),(0,0),(0,0),(0,0)]
		self.decrease = self.times/MAX_FRAMES#7000000
		print('Number of times: ', self.times)
		return (allies+enemies)

	def sync_update():
		pass

def main():
	sc = SimController()
	while(True):
		sc.sync_update()


if __name__ == '__main__':
	main()