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
from time import sleep, time
import sys
import math

#TODO aceleracao em vez de velocidades
#trablahar nos hiperparametros
#aumentar recompensas vs recompensas negativas
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
GAMMA = 0.4
MAX_FRAMES = 80000
ALPHA = 0.7
EPSILON = 1	#change for 0.1


MAX_LIN_SPEED = 50
MIN_LIN_SPEED = 0
LIN_STEP = 5
MAX_ANG_SPEED = 6
MIN_ANG_SPEED = -6
ANG_STEP = 1

MAX_LIN_ACCEL = 5
MAX_ANG_ACCEL = 2

MAX_MEMORY_SPEED = 20

#NUMBER_OF_ACTIONS = ((MAX_LIN_SPEED - MIN_LIN_SPEED + LIN_STEP)//LIN_STEP)*((MAX_ANG_SPEED -MIN_ANG_SPEED+ANG_STEP)//ANG_STEP)
# NB_LIN_ACT = (MAX_LIN_ACCEL*2+1)
# NB_ANG_ACT = (MAX_ANG_ACCEL*2 +1)
# NUMBER_OF_ACTIONS = NB_ANG_ACT*NB_LIN_ACT
NUMBER_OF_ACTIONS = 9

LIN_SPEED_VEC = [50, 0, -30]
ANG_SPEED_VEC = [3, 0, -3]

NUMBER_OF_PLAYERS = 1


BALL_MAX_X = 76
BALL_MIN_X = -76
FEATURE_PLAYER = 3#8
NUMBER_BALL_FEATURES = 6#5
NUM_FEATURES = NUMBER_BALL_FEATURES + FEATURE_PLAYER*NUMBER_OF_PLAYERS

LIN_ACCEL_VEC = [i for i in range(-5,6)]
ANG_ACCEL_VEC = [i for i in range(-2, 3)]

MAX_FRAMES_GAME = 600

MAX_ANGLE_FRONT = 0.62

OBSERVE_TIMES = 100#1800#3600 # BUFFER #com 100 funcionou legal
MAX_MEMORY_BALL = 30

MAX_EPISODES = 50


BATCH_SIZE = 4*OBSERVE_TIMES//6 #OBSERVE_TIMES #1000


MAX_DIST = 152

TAU = 0.01

model_name = 'mymodel_episodic_v2.h5'
MIN_DELTA_NO_MOVEMENT = 0.5
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def distance_cool(angle1, angle2):
	v1 = [math.cos(angle1), math.sin(angle1)]
	v2 = [math.cos(angle2), math.sin(angle2)]
	if(v1 == v2):
		return 0
	return angle(v1, v2)
def distance_between_bodies(body1, body2):
	return math.sqrt((body1.position[0] - body2.position[0])**2  + (body1.position[1] - body2.position[1])**2)

def distance_between_ball_and_goal(body1):
	return math.sqrt((body1.position[0] - 0)**2  + (body1.position[1] - 76)**2)
#transform an input of robot_allies, robot_opponents, and ball to a valid array
def angle_between_bodies(body1, body2):
	dx = body2.position[0] - body1.position[0]
	dy = body2.position[1] - body1.position[1]
	if(dx == 0):
		return (math.pi/2)
	if(dy == 0):
		return 0
	number = math.atan(dy/dx)
	if(dx > 0 and dy > 0):
		return number
	elif(dx > 0 and dy < 0):
		return math.pi*2 + number
	elif(dx < 0 and dy > 0):
		return (number + math.pi)
	elif(dx < 0 and dy < 0):
		return math.pi + number

	return math.atan(())
def scale(number, max_s, angle=False):
	if(angle):
		if(number > 2*math.pi):
			number -= 2*math.pi
	return round(((number/max_s)), 2)
def transform_to_state(robot_allies, robot_opponents, ball, inputs=None):
	state = []
	state.append(scale(ball.body.position[0], 76))
	state.append(scale(ball.body.position[1], 60))
	# state.append(scale(ball.body.linearVelocity[0], 50))
	# state.append(scale(ball.body.linearVelocity[1], 50))
	state.append(scale(distance_between_ball_and_goal(ball.body),152))
	for a in range(NUMBER_OF_PLAYERS):
		state.append(scale(robot_allies[a].body.position[0],76))
		state.append(scale(robot_allies[a].body.position[1],60))
		state.append(scale(robot_allies[a].body.angle,2*math.pi))
		# state.append(scale(robot_allies[a].body.linearVelocity[0],50))
		# state.append(scale(robot_allies[a].body.linearVelocity[1],50))
		# state.append(scale(robot_allies[a].body.angularVelocity, 6))
		state.append(scale(distance_between_bodies(robot_allies[a].body, ball.body), 152))
		state.append(scale(angle_between_bodies(robot_allies[a].body, ball.body),2*math.pi))
		#state.append(scale(distance_cool(robot_allies[a].body.angle, angle_between_bodies(robot_allies[a].body, ball.body)),math.pi))
		
		# state.append(distance_between_bodies(robot_allies[a].body, ball.body))
		# state.append(angle_between_bodies(robot_allies[a].body, ball.body))
		state.append(distance_cool(robot_allies[a].body.angle, angle_between_bodies(robot_allies[a].body, ball.body)))
	#print('distance between: ', distance_between_bodies(robot_allies[0].body, ball.body))
	#print('angle between: ', math.degrees(angle_between_bodies(robot_allies[a].body, ball.body)))
	state = np.array(state)
	state = state.reshape(NUM_FEATURES,1)
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
		self.isGoal = False
		self.restart = False
		self.times = 1
		self.iterations = 0
		self.replay_memory = []
		self.old_state = None
		self.action_space = []
		self.action_number_ang = None
		self.action_number_lin = None
		self.times_since_restart = 1
		self.episodes = 0
		# for angle in range(MIN_ANG_SPEED, MAX_ANG_SPEED+ ANG_STEP, ANG_STEP):
		# 	for linear in range(MIN_LIN_SPEED, MAX_LIN_SPEED +LIN_STEP, LIN_STEP):
		# 		self.action_space.append((angle, linear))
		for angle in ANG_SPEED_VEC:
			for linear in LIN_SPEED_VEC:
				self.action_space.append((angle, linear))
		# for angle in range(-MAX_ANG_ACCEL, MAX_ANG_ACCEL+ 1):
		# 	for linear in range(-MAX_LIN_ACCEL, MAX_LIN_ACCEL +1):
		# 		self.action_space.append((angle, linear))
		self.speed = [0,0]

		self.action = 0
		if(only_play or load_model):
			self.model = nn.neural_net_model2_2(NUMBER_OF_PLAYERS, model_name)
			self.target_model = nn.neural_net_model2_2(NUMBER_OF_PLAYERS, model_name)
		else :
			self.model = nn.neural_net_model2_2(NUMBER_OF_PLAYERS)
			self.target_model = nn.neural_net_model2_2(NUMBER_OF_PLAYERS)
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
		return
		if(len(self.ball_memory) >= MAX_MEMORY_BALL):
			self.ball_memory.popleft()
		self.ball_memory.append((ball.body.position[0], ball.body.position[1]))

	def add_player_memory(self, player):
		return
		if(len(self.player_memory) >= MAX_MEMORY_BALL):
			self.player_memory.popleft()
		self.player_memory.append((player.body.position[0], player.body.position[1], player.body.angle))
	#TODO
	def getReward(self, new_state, robot_allies, robot_opponents, ball):
		reward = -1 #- self.t_hits//30
		diff_direction = distance_cool(robot_allies[0].body.angle, angle_between_bodies(robot_allies[0].body, ball.body))
		diff_distance = distance_between_bodies(robot_allies[0].body, ball.body)
		# if(diff_direction < 0.01):
		# 	diff_direction = 0.01
		if(diff_distance < 1):
			diff_distance = 1 
		# print('diff_distance',diff_distance)
		# print('diff direct', diff_direction)
		reward += (1/(0.1 + diff_direction) + 1/(0.05 + 0.01*diff_distance))/(0.1*self.times_since_restart)
		ball_x = ball.body.position[0]
		if(ball_x <= BALL_MIN_X):
			reward = self.reward['enemy_goal']
			self.restart = True
		elif(ball_x >= BALL_MAX_X):
			self.isGoal = True
			reward = self.reward['goal']
			self.restart = True
			print('goall!!!!')

		elif(self.isPlayerStuck()):
			print('stuck')
			reward -= 30
			# self.restart = True
			self.player_memory = deque()
		elif(self.playerHitBall(robot_allies, robot_opponents, ball)):
			self.t_hits = 0
			reward += 200
		elif(self.times%MAX_FRAMES_GAME == 0 and ball_x < BALL_MAX_X):
			# self.restart = True
			reward = -300
		elif(self.isSpinning()):
			print('stuck spinning')
			reward = reward - 30
			self.last_speeds = deque()
		self.t_hits+=1
		self.times_since_restart +=1
		if(self.restart):
			self.times_since_restart = 1
		# if(not self.isBallMoving()):
		# 	reward = reward - 10
		# 	self.ball_memory = deque()
		#reward += 1000/(distance_between_bodies(robot_allies[0].body, ball.body))
		#print('reward',reward)
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
	def target_train(self):
		# model_weights = self.model.get_weights()
		# target_model_weights = self.target_model.get_weights()
		# weight_list = np.arange(len(model_weights))
		# target_model_weights[weight_list] = TAU * model_weights[weight_list] + (1 - TAU) * target_model_weights[weight_list]
		# self.target_model.set_weights(target_model_weights)
		model_weights = self.model.get_weights()
		target_model_weights = self.target_model.get_weights()
		weight_list = np.arange(len(model_weights))
		target_model_weights[weight_list] = TAU * model_weights[weight_list] + (1 - TAU) * target_model_weights[weight_list]
		self.target_model.set_weights(target_model_weights)


	def isPlayerStuck(self):
		if(len(self.player_memory) < MAX_MEMORY_BALL):
			return False
		# x = list(map(lambda x : x[0], self.player_memory))
		# y = list(map(lambda x : x[1], self.player_memory))
		dx = abs(self.player_memory[0][0] - self.player_memory[MAX_MEMORY_BALL-1][0])
		dy = abs(self.player_memory[0][1] - self.player_memory[MAX_MEMORY_BALL-1][1])
		dangle = abs(self.player_memory[MAX_MEMORY_BALL-5][2] - self.player_memory[MAX_MEMORY_BALL-1][0])
		if(dx < 0.1 and dy < 0.1 and dangle < 0.1):
			return True
		return False

	def playerHitBall(self, robot_allies, robot_opponents, ball):
		if(robot_allies[0].body.userData == ball.body):
			robot_allies[0].body.userData = None
			if(distance_cool(robot_allies[0].body.angle, angle_between_bodies(robot_allies[0].body, ball.body)) < MAX_ANGLE_FRONT):
				print('hit!')
				return True
		return False


	def compute(self, robot_allies, robot_opponents, ball):
		if(self.times%4 != 0 and not only_play):
			self.times+=1
			return
		new_state = transform_to_state(robot_allies, robot_opponents, ball, [self.action_number_ang, self.action_number_lin])
		reward = self.getReward(new_state, robot_allies, robot_opponents, ball)

		#print('reward: ',reward)
		if(len(self.replay_memory) < OBSERVE_TIMES-1 and (not self.restart)):
			self.replay_memory.append((self.old_state[:], self.action, reward, new_state[:]))
		else :
			self.replay_memory.append((self.old_state[:], self.action, reward, new_state[:]))
			#exit()
			#build batch
			#batch = random.sample(self.replay_memory, BATCH_SIZE)
			if(len(self.replay_memory) < BATCH_SIZE):
				batch = self.replay_memory
			else:
				batch = random.sample(self.replay_memory,BATCH_SIZE)
			t1 = time()
			x_train, y_train = self.generate_train_from_batch2(batch)
			elapsed = time() - t1
			print('time to batch: ', elapsed*1000, 'ms')
			#x_train = np.expand_dims(x_train, axis=2)
			print('fitting...')
			self.replay_memory = []
			t1 = time()
			# self.model.fit(x_train, [y_train,y2_train], batch_size=BATCH_SIZE, epochs=5, verbose=0)
			#self.model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=20, verbose=0)
			self.model.train_on_batch(x_train, y_train)
			elapsed = time() - t1
			print('time to fit: ', elapsed*1000, 'ms')
		self.add_ball_memory(ball)
		self.add_player_memory(robot_allies[0])
		if(self.times_since_restart%MAX_FRAMES_GAME==0 or self.restart):
			print('saving and restarting...')
			print(self.episodes, 'out of ', MAX_EPISODES*10)
			self.episodes+=1
			self.restart = True
			self.model.save_weights(model_name)
		if(self.restart):
			self.treatRestart()
		self.times+=1
		if(self.episodes > MAX_EPISODES*10):#self.times > MAX_FRAMES*10):
			print('saving and exiting ...')
			self.model.save(model_name)
			exit()

	def treatRestart(self):
		self.times_since_restart = 1
		self.replay_memory = []
		self.last_speeds = deque()
		self.ball_memory = deque()
		self.player_memory = deque()

	def isTerminalState(self,reward):
		if(reward == -300 or reward == -100 or reward == self.reward['goal'] or reward == self.reward['enemy_goal']):
			return True
		return False

	def generate_train_from_batch2(self, batch):
		x_train = []
		decay = ALPHA
		mb_len = len(batch)
		old_states = np.zeros(shape=(mb_len,NUM_FEATURES))
		rewards = np.zeros(shape=(mb_len,))
		actions = np.zeros(shape=(mb_len,NUMBER_OF_PLAYERS,))
		new_states = np.zeros(shape=(mb_len, NUM_FEATURES))
		d_actions = np.zeros(shape=(mb_len,))
		for i, memory in enumerate(batch):
			old_state_m, action, reward_m, new_state_m = memory
			old_states[i, :] = old_state_m.transpose()[...]
			actions[i] = action
			rewards[i] = reward_m
			new_states[i, :] = new_state_m.transpose()[...]
			d_actions[i] = self.isTerminalState(reward_m)

		old_qvals = self.model.predict(old_states, batch_size=mb_len)

		new_qvals = self.target_model.predict(new_states, batch_size=mb_len)

		maxQs = np.max(new_qvals, axis=1)

		terminal_idx = np.where(d_actions == 1)[0]
		non_term_idx = np.where(d_actions==0)[0]
		#maxQs = np.array(maxQs)
		# maxQs = np.max(new_qvals[0], axis=1)
		y = old_qvals[:]
		batch_list = np.linspace(0,mb_len-1,mb_len, dtype=int)
		#y[batch_list, actions[batch_list,0].astype(int)] = (1-ALPHA)*y[batch_list, actions[batch_list,0].astype(int)] + ALPHA*(rewards[batch_list] + (GAMMA * maxQs[batch_list]))
		y[terminal_idx, actions[terminal_idx,0].astype(int)] = (rewards[terminal_idx])
		y[non_term_idx, actions[non_term_idx,0].astype(int)] = (rewards[non_term_idx] + (GAMMA * maxQs[non_term_idx]))
		
		X_train = old_states
		return X_train, y


	def generate_train_from_batch(self, batch):
		x_train = []
		y_train = []
		y2_train = []
		for memory in batch:
			old_state, action_number_ang, action_number_lin, reward, new_state = memory
			old_qval = self.model.predict(old_state.transpose())
			new_qval = self.model.predict(new_state.transpose())
			# old_qval = self.model.predict(np.expand_dims(old_state.transpose(),axis=2))
			# new_qval = self.model.predict(np.expand_dims(new_state.transpose(),axis=2))
			#max_qval = np.max(new_qval)
			max_qval_ang = np.max(new_qval[0])
			max_qval_lin = np.max(new_qval[1])
			if(not self.isTerminalState(reward)):
				update = reward + GAMMA*max_qval_ang
				update2 = reward + GAMMA*max_qval_lin
			else :
				update = reward
				update2 = reward
			#decay = max((ALPHA - self.times/(MAX_FRAMES*6)), 0.15)
			decay = ALPHA
			y = []
			y[:] = old_qval[0][:]
			old_qval_action = y[0][action_number_ang]
			y[0][action_number_ang] = (1-decay)*old_qval_action + decay*update

			y2 = []
			y2[:] = old_qval[1][:]
			old_qval_action = y2[0][action_number_lin]
			print(y2[0][action_number_lin])
			y2[0][action_number_lin] = (1-decay)*old_qval_action + decay*update2
			print(y2[0][action_number_lin])
			y = np.array(y)
			y2 = np.array(y2)
			x_train.append(old_state.reshape((NUM_FEATURES),))
			#y_train.append(y.reshape(NUMBER_OF_ACTIONS,))
			y_train.append(y.reshape(NB_ANG_ACT))
			y2_train.append(y2.reshape(NB_LIN_ACT))
		x_train = np.array(x_train)
		y_train = np.array(y_train)
		y2_train = np.array(y2_train)
		return x_train, y_train, y2_train


	def action_saturate(self, act):
		angle_v, lin_v = act
		lin_v = min(max(MIN_LIN_SPEED, lin_v), MAX_LIN_SPEED)
		angle_v = min(max(MIN_ANG_SPEED, angle_v), MAX_ANG_SPEED)
		return [angle_v, lin_v]


	def sync_control_centrallized(self, ally_positions, enemy_positions, ball):
		if(only_play):
			self.times +=1
		if(self.times%4!=0 and not only_play):
			return
		lastinputs = [self.action_number_ang, self.action_number_lin]
		state = transform_to_state(ally_positions, enemy_positions, ball, lastinputs)
		self.old_state = state[:]
		#print('state',state.transpose())
		dec = max(EPSILON - self.decrease, 0.08)
		#print('EPSILON', dec)
		if((random.random() < dec or self.times < OBSERVE_TIMES) and not only_play):
			# action_ang = random.randint(0, (len(ANG_ACCEL_VEC)-1))
			# action_lin = random.randint(0, (len(LIN_ACCEL_VEC)-1))
			# action_ang = random.randint(0, (len(ANG_SPEED_VEC)-1))
			# action_lin = random.randint(0, (len(LIN_SPEED_VEC)-1))
			action = (random.randint(0,NUMBER_OF_ACTIONS-1))
		else :
			predicted_qval = self.model.predict(state[:].transpose(), batch_size=1) #checar batch size!!
			#predicted_qval = self.model.predict(np.expand_dims(state.transpose(),axis=2), batch_size=1) #checar batch size!!
			action = np.argmax(predicted_qval)
			# print('val:', predicted_qval)
		#self.speed = [ANG_SPEED_VEC[action_ang],LIN_SPEED_VEC[action_lin]]
		# ang_accel = ANG_ACCEL_VEC[action_ang]
		# lin_accel = LIN_ACCEL_VEC[action_lin]
		self.speed = self.action_space[action]
		self.action = action
		# self.action = [ang_accel, lin_accel]
		# #print('action: ', self.action)
		# self.speed[0] += self.action[0]
		# self.speed[1] += self.action[1]
		# self.speed = self.action_saturate(self.speed)
		self.add_speed_memory(self.speed)
		#print('speed: ', self.speed)
		(a,b) = self.speed
		allies = [(a,b),(0,0),(0,0),(0,0),(0,0)]
		enemies = [(0,0),(0,0),(0,0),(0,0),(0,0)]
		#self.decrease = self.times/MAX_FRAMES#7000000
		self.decrease = 10*self.episodes/(MAX_EPISODES)#*3)
		# print(self.times, 'out of ', MAX_FRAMES*10)
		return (allies+enemies)


def main():
	sc = SimController()
	while(True):
		sc.sync_update()


if __name__ == '__main__':
	main()