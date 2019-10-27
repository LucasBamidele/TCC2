"""
TODO:
	Create temporal series for states (define size)
		maybe try to predict, need to work this better
	expand to 3 vs 3 players (invert input before)
	try centralized approach first
	work on a better model for cnn and basic ann
	
"""
seed = 4001
import numpy as np
from NeuralNet3 import *
import random
# from collections import deque
# from time import sleep, time
# import sys
# import math
# import LogAndPlot as plotter

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.distributions import Normal

from keras.utils import to_categorical
from collections import deque
from time import sleep, time
import sys
import math

#seeding
np.random.seed(seed)
random.seed(seed)
# set_random_seed(seed)
#TODO aceleracao em vez de velocidades
#trablahar nos hiperparametros
#aumentar recompensas vs recompensas negativas
load_model = False
only_play = False
p = None
if(len(sys.argv)>1):
    if(sys.argv[1]=='play'):
        only_play = True
    if(sys.argv[1]=='load' or sys.argv[1]=='load_no_window'):
    	load_model = True
if(len(sys.argv) > 2):
	p = sys.argv[2]



GOAL_REWARD = 1200
PASS_REWARD = 100
RETAKE_REWARD = 100
ENEMY_GOAL_REWARD = -100
STUCK_REWARD = -100
HIT_REWARD = 100
TIMEOUT_REWARD = 0

GAMMA = 0.93
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

MAX_MEMORY_SPEED = 40

#NUMBER_OF_ACTIONS = ((MAX_LIN_SPEED - MIN_LIN_SPEED + LIN_STEP)//LIN_STEP)*((MAX_ANG_SPEED -MIN_ANG_SPEED+ANG_STEP)//ANG_STEP)
# NB_LIN_ACT = (MAX_LIN_ACCEL*2+1)
# NB_ANG_ACT = (MAX_ANG_ACCEL*2 +1)
# NUMBER_OF_ACTIONS = NB_ANG_ACT*NB_LIN_ACT
NUMBER_OF_ACTIONS = 9
if(only_play):
	LIN_SPEED_VEC = [50, 0, -50]
	ANG_SPEED_VEC = [5, 0, -5]
else:
	LIN_SPEED_VEC = [30, 0, -30]
	ANG_SPEED_VEC = [3, 0, -3]


NUMBER_OF_PLAYERS = 1

#PPO VARIABLES
C1 = 0.5
C2 = - 0.001
LAYER_SIZE = 128

BALL_MAX_X = 76
BALL_MIN_X = -76
FEATURE_PLAYER = 9#8
NUMBER_BALL_FEATURES = 6#5
NUM_FEATURES = NUMBER_BALL_FEATURES + FEATURE_PLAYER*NUMBER_OF_PLAYERS

LIN_ACCEL_VEC = [i for i in range(-5,6)]
ANG_ACCEL_VEC = [i for i in range(-2, 3)]

MAX_FRAMES_GAME = 390
BUFFER_SIZE = MAX_FRAMES_GAME
MAX_ANGLE_FRONT = 0.62

OBSERVE_TIMES = 200#1800#3600 # BUFFER #com 100 funcionou legal
MAX_MEMORY_BALL = 60

MAX_EPISODES = 3000

# use_cuda = torch.cuda.is_available()
# device   = torch.device("cuda" if use_cuda else "cpu")
EPOCHS = 10
TEST_CRITIC = 100

DUMMY_ACTION, DUMMY_VALUE, dummy_rewards, dummy_pred_values= np.zeros((1, NUMBER_OF_ACTIONS)), np.zeros((1, 1)),  np.zeros((1, 1)),  np.zeros((1, 1))


BATCH_SIZE = MAX_FRAMES_GAME #OBSERVE_TIMES #1000
END_GAME_THRESHOLD = MAX_FRAMES_GAME*1.5

MAX_DIST = 152

TAU = 0.4
if(p):
	p = str(p)
	file_name = 'mymodel_ppo3'
	model_name = p + file_name + '.h5'
	file_name_crit = p + 'crit' + file_name
	model_name2 = file_name_crit + '.h5'
	_dir = 'saved_models/'
else :
	file_name = 'mymodel_ppo3'
	model_name = file_name + '.h5'
	file_name_crit = 'crit' + file_name
	model_name2 = file_name_crit + '.h5'
	_dir = 'saved_models/'
MIN_DELTA_NO_MOVEMENT = 0.5
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def distance_cool(angle1, angle2):
	x1 = math.cos(angle1)
	x2 = math.cos(angle2)
	y1 = math.sin(angle1)
	y2 = math.sin(angle2)
	dot = x1*x2 + y1*y2      # dot product
	det = x1*y2 - y1*x2      # determinant
	# v1 = [math.cos(angle1), math.sin(angle1)]
	# v2 = [math.cos(angle2), math.sin(angle2)]

	# if(v1 == v2):
	# 	return 0
	#return angle(v1,v2)
	return np.arctan2(det,dot)
def distance_between_bodies(body1, body2):
	return math.sqrt((body1.position[0] - body2.position[0])**2  + (body1.position[1] - body2.position[1])**2)

def distance_between_ball_and_goal(body1, enemy=False):
	if(enemy):
		return abs(body1.position[0] + 76)
	return abs(body1.position[0] - 76)
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

def angle_between_body_and_goal(body1, enemy=False):
	bla= 76
	if(enemy):
		bla = -76
	dx = body1.position[0] - 0
	dy = body1.position[1] - bla
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
def scale2(number, max_s, min_s=0, angle=False):
	a =  2*(number - min_s)/(max_s-min_s) - 1
	return round(a, 2)
def transform_to_state(robot_allies, robot_opponents, ball, inputs=None, enemy=False):
	state = []
	state.append(scale(ball.body.position[0], 78))
	state.append(scale(ball.body.position[1], 61))
	state.append(scale2(angle_between_body_and_goal(ball.body, enemy), 152))
	state.append(scale(ball.body.linearVelocity[0], 50))
	state.append(scale(ball.body.linearVelocity[1], 50))
	state.append(scale2(distance_between_ball_and_goal(ball.body, enemy),152))
	for a in range(NUMBER_OF_PLAYERS):
		state.append(scale(robot_allies[a].body.position[0],76))
		state.append(scale(robot_allies[a].body.position[1],60))
		state.append(scale2(robot_allies[a].body.angle,2*math.pi))
		state.append(scale(robot_allies[a].body.linearVelocity[0],50))
		state.append(scale(robot_allies[a].body.linearVelocity[1],50))
		state.append(scale(robot_allies[a].body.angularVelocity, 6))
		state.append(scale2(distance_between_bodies(robot_allies[a].body, ball.body), 152))
		state.append(scale2(angle_between_bodies(robot_allies[a].body, ball.body),2*math.pi))
		# state.append(scale2(angle_between_body_and_goal(robot_allies[a].body, enemy), 152))
		#state.append(scale(distance_cool(robot_allies[a].body.angle, angle_between_bodies(robot_allies[a].body, ball.body)),math.pi))
		
		# state.append(distance_between_bodies(robot_allies[a].body, ball.body))
		# state.append(angle_between_bodies(robot_allies[a].body, ball.body))
		state.append(scale(distance_cool(robot_allies[a].body.angle, angle_between_bodies(robot_allies[a].body, ball.body)),math.pi))
	#print('distance between: ', distance_between_bodies(robot_allies[0].body, ball.body))
	#print('angle between: ', math.degrees(angle_between_bodies(robot_allies[a].body, ball.body)))
	state = np.array(state)
	# state = torch.FloatTensor(state).to(device)
	# state = state.unsqueeze(0)
	state = state.reshape(NUM_FEATURES,1)
	#exit()
	# for ally in robot_allies:
	# 	state.append((ally.body.position[0], ally.body.position[1]))
	# for oppon in robot_opponents:
	# 	state.append((oppon.body.position[0], oppon.body.position[1]))
	# state = np.array(state)
	return state

def create_action_space():
	fw = LIN_SPEED_VEC[0]
	bw = LIN_SPEED_VEC[2]
	r = ANG_SPEED_VEC[0]
	l = ANG_SPEED_VEC[2]
	N = (0, fw)
	NE = (r, fw)
	E = (r, 0)
	SE = (r, bw)
	S = (0, bw)
	SW = (l, bw)
	W = (l, 0)
	NW = (l, fw)
	STOP = (0,0)
	return [N, NE, E, SE, S, SW, W, NW, STOP]

class SimController(object):
	"""docstring for SimController"""
	def __init__(self, isEnemy = False):
		self.isEnemy = isEnemy
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
		self.latest_rewards = []
		self.mean_rewards = []
		self.log_loss = []
		self.log_loss_model = []
		self.printed = False
		# for angle in range(MIN_ANG_SPEED, MAX_ANG_SPEED+ ANG_STEP, ANG_STEP):
		# 	for linear in range(MIN_LIN_SPEED, MAX_LIN_SPEED +LIN_STEP, LIN_STEP):
		# 		self.action_space.append((angle, linear))
		# for angle in ANG_SPEED_VEC:
		# 	for linear in LIN_SPEED_VEC:
		# 		self.action_space.append((angle, linear))
		self.action_space = create_action_space()
		# for angle in range(-MAX_ANG_ACCEL, MAX_ANG_ACCEL+ 1):
		# 	for linear in range(-MAX_LIN_ACCEL, MAX_LIN_ACCEL +1):
		# 		self.action_space.append((angle, linear))
		self.speed = [0,0]

		self.action = 0
		if(only_play or load_model):
			self.critic = Critic(NUM_FEATURES, NUMBER_OF_ACTIONS, LAYER_SIZE, _dir + model_name2)
			self.actor = Actor(NUM_FEATURES, NUMBER_OF_ACTIONS, LAYER_SIZE, _dir + model_name)
		else :
			self.critic = Critic(NUM_FEATURES, NUMBER_OF_ACTIONS, LAYER_SIZE)
			self.actor = Actor(NUM_FEATURES, NUMBER_OF_ACTIONS, LAYER_SIZE)
		self.reward_dict = {
			'goal': GOAL_REWARD,
			'pass': PASS_REWARD,
			'retake': RETAKE_REWARD,
			'enemy_goal': ENEMY_GOAL_REWARD,
			'stuck': STUCK_REWARD,
			'hit': HIT_REWARD,
			'timeout' : TIMEOUT_REWARD
		}
		self.last_speeds = deque()
		self.ball_memory = deque()
		self.player_memory = deque()
		self.t_hits = 0
		self.timeout = False

		#ppo
		self.play = False
		self.val = False
		self.predicted_action = None
		self.dist = None
		self.value = None
		self.entropy = 0
		self.action_matrix = None
		self.batch = [[], [], [], [], []]
		self.tmp_batch = [[], [], [], []]
		super(SimController, self).__init__()


	def compute_gae(self, next_value, rewards, masks, values, gamma=GAMMA, tau=0.95):
		# values = values + [next_value]
		values = np.append(values, next_value, 0)
		gae = 0
		returns = []
		for step in reversed(range(len(rewards))):
			delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
			gae = delta + gamma * tau * masks[step] * gae
			returns.insert(0, gae + values[step])
		return returns

	def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
		batch_size = states.size(0)
		for _ in range(batch_size // mini_batch_size):
			rand_ids = np.random.randint(0, batch_size, mini_batch_size)
			yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
	        
	        

	def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
		for _ in range(ppo_epochs):
			for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
				dist, value = model(state)
				entropy = dist.entropy().mean()
				new_log_probs = dist.log_prob(action)

				ratio = (new_log_probs - old_log_probs).exp()
				surr1 = ratio * advantage
				surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

				actor_loss  = - torch.min(surr1, surr2).mean()
				critic_loss = (return_ - value).pow(2).mean()

				loss = C1 * critic_loss + actor_loss + C2 * entropy

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

	def add_speed_memory(self, speed):
		return
		if(len(self.last_speeds) >= MAX_MEMORY_SPEED):
			self.last_speeds.popleft()
		self.last_speeds.append(speed[:])

	def add_ball_memory(self,ball):
		return
		if(len(self.ball_memory) >= MAX_MEMORY_BALL):
			self.ball_memory.popleft()
		self.ball_memory.append((ball.body.position[0], ball.body.position[1]))

	def add_player_memory(self, player):
		if(len(self.player_memory) >= MAX_MEMORY_BALL):
			self.player_memory.popleft()
		self.player_memory.append((player.body.position[0], player.body.position[1], player.body.angle))
	
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

	def isPlayerStuck(self):
		if(len(self.player_memory) < MAX_MEMORY_BALL):
			return False
		# x = list(map(lambda x : x[0], self.player_memory))
		# y = list(map(lambda x : x[1], self.player_memory))
		dx = abs(self.player_memory[0][0] - self.player_memory[MAX_MEMORY_BALL-1][0])
		dy = abs(self.player_memory[0][1] - self.player_memory[MAX_MEMORY_BALL-1][1])
		dangle = abs(self.player_memory[MAX_MEMORY_BALL-5][2] - self.player_memory[MAX_MEMORY_BALL-1][2])
		if(dx < 0.1 and dy < 0.1 and dangle < 0.1):
			self.isStuck = True
			self.times_stuck = 0
			return True
		return False

	def playerHitBall(self, robot_allies, robot_opponents, ball):
		if(robot_allies[0].body.userData == ball.body):
			robot_allies[0].body.userData = None
			if(abs(distance_cool(robot_allies[0].body.angle, angle_between_bodies(robot_allies[0].body, ball.body))) < MAX_ANGLE_FRONT):
				print('hit!')
				self.t_hits = self.reward_dict['hit']*GAMMA
				return True
		return False

	def treatRestart(self):
		self.isGoal = False
		# self.times_since_restart = 1
		self.replay_memory = []
		self.last_speeds = deque()
		self.ball_memory = deque()
		self.player_memory = deque()
		self.episodes+=1
		self.printed = False
		self.play = False
		if(self.episodes%MAX_EPISODES==(MAX_EPISODES-1)):
			print('saving...')
			enemy_sv = ''
			if(self.isEnemy):
				enemy_sv = 'enemy-'
			self.actor.save_weights(_dir + str(self.episodes) + enemy_sv +model_name)
			self.critic.save_weights(_dir + str(self.episodes) + enemy_sv + model_name2)
		# self.latest_rewards = []

	def isTerminalState(self,reward):
		if(reward == self.reward_dict['timeout'] or reward == self.reward_dict['goal'] or reward == self.reward_dict['enemy_goal']):
			return True
		if(self.timeout):
			return True
		# if(self.times_since_restart > BUFFER_SIZE):
		# 	return True
		return False



	#TODO
	def getReward(self, new_state, robot_allies, robot_opponents, ball):
		reward = 0 #- self.t_hits//30
		diff_ball = distance_between_ball_and_goal(ball.body, self.isEnemy)
		diff_direction = abs(distance_cool(robot_allies[0].body.angle, angle_between_bodies(robot_allies[0].body, ball.body)))
		diff_distance = distance_between_bodies(robot_allies[0].body, ball.body)
		if(diff_direction < 0.01):
			diff_direction = 0.01
		if(diff_distance < 1):
			diff_distance = 1 
		# print('diff_distance',diff_distance)
		# print('diff direct', diff_direction)
		# reward1 = 1/(0.05 + 0.01*diff_distance)
		# print('reward1', reward1)
		reward += 5/(0.1 + 0.1*diff_direction) + 5/(0.05 + 0.01*diff_distance)#/(0.1*self.times_since_restart)
		# print('reward',reward)
		# reward +=reward1
		goal_x = MAX_DIST/2
		reward += 1000*((goal_x- diff_ball)/goal_x)
		ball_x = ball.body.position[0]
		reward += self.t_hits
		if(ball_x <= BALL_MIN_X):
			reward = self.reward_dict['enemy_goal']
			self.restart = True
			self.isGoal = True
			if(self.isEnemy):
				reward = self.reward_dict['goal']
				self.restart = True
				self.isGoal = False
				print('enemy goal!!')
		elif(ball_x >= BALL_MAX_X):
			self.isGoal = True
			reward = self.reward_dict['goal']
			self.restart = True
			print('ally goall!!!!')
			if(self.isEnemy):
				self.isGoal = False
				reward = self.reward_dict['enemy_goal']
				self.restart = True
				print('ally goall!!!!')
		elif(self.isPlayerStuck()):
			print('stuck!!')
			#reward -= 30
			# reward = self.reward_dict['stuck']
			# self.restart = True
			self.player_memory = deque()
		elif(self.playerHitBall(robot_allies, robot_opponents, ball)):
			reward += self.reward_dict['hit']
		# elif(self.times%MAX_FRAMES_GAME == 0 and ball_x < BALL_MAX_X):
		# 	#self.restart = True
		# 	reward = -300 + ball_x
		# elif(self.isSpinning()):
		# 	print('stuck spinning')
		# 	reward = reward - 30
		# 	self.last_speeds = deque()
		elif(self.times_since_restart >= END_GAME_THRESHOLD):
			self.restart = True
			# reward = self.reward_dict['timeout']
			self.timeout = True
			print('timeout!')
		if(self.t_hits > 1):
			self.t_hits = GAMMA*self.t_hits
		else:
			self.t_hits = 0
		self.times_since_restart +=1
		# if(not self.isBallMoving()):
		# 	reward = reward - 10
		# 	self.ball_memory = deque()
		#reward += 1000/(distance_between_bodies(robot_allies[0].body, ball.body))
		# print('reward',reward)
		#evaluate the reward from the game state!
		return reward

		#x.pop(left)
	def target_train(self):
		model_weights = np.array(self.model.get_weights())
		target_model_weights = np.array(self.target_model.get_weights())
		weight_list = np.arange(len(model_weights))
		target_model_weights[weight_list] = TAU * model_weights[weight_list] + (1 - TAU) * target_model_weights[weight_list]
		self.target_model.set_weights(target_model_weights)
		# model_weights = self.model.get_weights()
		# target_model_weights = self.target_model.get_weights()
		# #weight_list = np.arange(len(model_weights))
		# for i in range(len(model_weights)):
		# 	target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
		# self.target_model.set_weights(target_model_weights)

	def transform_reward(self):
		for j in range(len(self.latest_rewards) - 2, -1, -1):
			self.latest_rewards[j] += self.latest_rewards[j + 1] * GAMMA

	def compute(self, robot_allies, robot_opponents, ball):
		if(self.times%4 != 0 and not only_play):
			self.times+=1
			return
		self.add_ball_memory(ball)
		self.add_player_memory(robot_allies[0])

		new_state = transform_to_state(robot_allies, robot_opponents, ball, [self.action_number_ang, self.action_number_lin])
		reward = self.getReward(new_state, robot_allies, robot_opponents, ball)
		self.latest_rewards.append(reward)

		done = self.isTerminalState(reward)

		self.tmp_batch[0].append(self.old_state[:].transpose())
		self.tmp_batch[1].append(self.action_matrix[:])
		self.tmp_batch[2].append(self.predicted_action[:])
		self.tmp_batch[3].append(1)

		if(done):
			# self.transform_reward() #what
			# if(self.val is False):
			for i in range(len(self.tmp_batch[0])):
				obs, action, prediction = self.tmp_batch[0][i][:], self.tmp_batch[1][i][:], self.tmp_batch[2][i][:]
				mask = self.tmp_batch[3][i]
				r = self.latest_rewards[i]
				self.batch[0].append(obs[:])
				self.batch[1].append(action[:])
				self.batch[2].append(prediction[:])
				self.batch[3].append(r)
				self.batch[4].append(mask)
			self.tmp_batch = [[], [], [], []]
		if(len(self.batch[0]) >= BUFFER_SIZE or self.isGoal):
			self.timeout = False
			self.times_since_restart = 1
			obs, action, pred, reward, mask = np.array(self.batch[0]), np.array(self.batch[1]), np.array(self.batch[2]), np.reshape(np.array(self.batch[3]), (len(self.batch[3]), 1)), np.array(self.batch[4])
			pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
			# obs, action, pred, reward = obs[-BUFFER_SIZE:], action[-BUFFER_SIZE:], pred[-BUFFER_SIZE:], reward[-BUFFER_SIZE:]
			# mask = mask[-BUFFER_SIZE:]
			b_size = len(obs)
			obs = np.reshape(obs, (b_size,NUM_FEATURES))
			old_prediction = pred
			pred_values = self.critic.predict(obs)
			next_value = self.critic.predict(new_state.transpose())
			returns_ = np.array(self.compute_gae(next_value, reward, mask, pred_values))
			print(returns_[-50:].transpose())
			print(pred_values[-50:].transpose())
			advantage = returns_ - pred_values
			advantage_norm = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-10)
			critic_loss = self.critic.fit([obs], [returns_], batch_size=b_size, shuffle=False, epochs=EPOCHS*3, verbose=False)
			print('critic loss: ', critic_loss.history['loss'][-1])
			actor_loss = self.actor.fit([obs, advantage_norm, old_prediction, returns_, pred_values], [action], batch_size=b_size, shuffle=False, epochs=EPOCHS, verbose=False)
			print('actor loss: ', actor_loss.history['loss'][-1])
			# for _ in range(EPOCHS*3):
			# 	critic_loss = self.critic.train_on_batch([obs], [returns_])
			# print('critic loss: ', critic_loss.history['loss'][-1])
			# for _ in range(EPOCHS):
			# 	actor_loss = self.actor.train_on_batch([obs, advantage, old_prediction], [action])
			# print('actor loss: ', actor_loss.history['loss'][-1])

			self.batch = [[], [], [], [], []]
			# self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
			# self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)

			# self.gradient_steps += 1
			self.latest_rewards = []


			self.mean_rewards.append(np.mean(self.latest_rewards[:]))
			print('restarting...')
			self.restart = True
			print(self.episodes, 'out of ', MAX_EPISODES*10)

		# else :
		# 	self.val = False
		# 	self.restart = True
			
		if(self.restart):
			self.treatRestart()
		self.times+=1
		if(self.episodes > MAX_EPISODES*10):#self.times > MAX_FRAMES*10):
			print('saving and exiting ...')
			enemy_sv = ''
			if(self.isEnemy):
				enemy_sv = 'enemy-'
			self.actor.save(enemy_sv + model_name)
			self.critic.save_weights(enemy_sv + file_name_crit)

			# plotter.plotRewards(self.mean_rewards, figname=file_name)
			exit()


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
		non_term_idx = np.where(d_actions == 0)[0]
		y = old_qvals[:]
		batch_list = np.linspace(0,mb_len-1,mb_len, dtype=int)
		y[terminal_idx, actions[terminal_idx,0].astype(int)] = (rewards[terminal_idx])
		y[non_term_idx, actions[non_term_idx,0].astype(int)] = (rewards[non_term_idx] + (GAMMA * maxQs[non_term_idx]))
		
		X_train = old_states
		return X_train, y

	def action_saturate(self, act):
		angle_v, lin_v = act
		lin_v = min(max(MIN_LIN_SPEED, lin_v), MAX_LIN_SPEED)
		angle_v = min(max(MIN_ANG_SPEED, angle_v), MAX_ANG_SPEED)
		return [angle_v, lin_v]


	def sync_control_centrallized(self, ally_positions, enemy_positions, ball):
		if(only_play):
			self.times +=1
		if(self.times%4!=0):# and not only_play):
			return
		lastinputs = [self.action_number_ang, self.action_number_lin]
		state = transform_to_state(ally_positions, enemy_positions, ball, lastinputs)
		self.old_state = state[:]
		predicted_qval = self.actor.predict([state[:].reshape(1,NUM_FEATURES), DUMMY_VALUE, DUMMY_ACTION,dummy_rewards, dummy_pred_values]) #checar batch size!!
		if(self.episodes%100==0):
			print(predicted_qval)
		if(self.episodes%10==0 and self.episodes > 100):
			self.play = True
		predicted_action = np.argmax(predicted_qval)
		self.predicted_action = predicted_qval
		# print('state',state.transpose())
		# p = random.random()
		action = np.random.choice(NUMBER_OF_ACTIONS, p=predicted_qval[0, :])
		if(only_play or self.play):
			action = predicted_action
		# if((not self.val or p < 0.15) and not only_play):
		# 	action = (random.randint(0,NUMBER_OF_ACTIONS-1))
		# else :
		# 	action = predicted_action
		self.action_matrix = np.zeros(NUMBER_OF_ACTIONS)
		self.action_matrix[action] = 1
		self.action = action
		self.speed = self.action_space[action]
		# self.action = action
		# #print('action: ', self.action)
		self.add_speed_memory(self.speed)
		#print('speed: ', self.speed)
		(a,b) = self.speed
		# print('0 OUTPUT')
		# a,b = 0,0
		allies = [(a,b),(0,0),(0,0),(0,0),(0,0)]
		enemies = [(0,0),(0,0),(0,0),(0,0),(0,0)]
		if(self.isEnemy):
			return (enemies+allies)
		# self.decrease = self.episodes/(2*MAX_EPISODES)
		return (allies+enemies)


def main():
	sc = SimController()
	while(True):
		sc.sync_update()


if __name__ == '__main__':
	main()