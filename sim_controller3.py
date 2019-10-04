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
GAMMA = 0.9
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
NB_LIN_ACT = (MAX_LIN_ACCEL*2+1)
NB_ANG_ACT = (MAX_ANG_ACCEL*2 +1)
NUMBER_OF_ACTIONS = NB_ANG_ACT*NB_LIN_ACT

NUMBER_OF_PLAYERS = 3
NUMBER_OF_ENEMIES = 3


BALL_MAX_X = 76
BALL_MIN_X = -76
FEATURE_PLAYER = 8 #3 
NUMBER_BALL_FEATURES = 5 #2
NUM_FEATURES = NUMBER_BALL_FEATURES + FEATURE_PLAYER*NUMBER_OF_PLAYERS + FEATURE_PLAYER*NUMBER_OF_ENEMIES

LIN_ACCEL_VEC = [i for i in range(-5,6)]
ANG_ACCEL_VEC = [i for i in range(-2, 3)]

MAX_FRAMES_GAME = 900

OBSERVE_TIMES = 3600#3600 # BUFFER
MAX_MEMORY_BALL = 15

MAX_EPISODES = 1000


BATCH_SIZE = OBSERVE_TIMES #OBSERVE_TIMES #1000


MAX_DIST = 152

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
def angle_between_bodies(body1, body2, angle=False):
	dx = body2.position[0] - body1.position[0]
	dy = body2.position[1] - body1.position[1]
	if(angle):
		dx = -dx
		dy = -dy
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
def scale(number, max_s,angle=False):
	if(angle):
		if(number > 2*math.pi):
			print(number)
			number -= 2*math.pi
			print(number)
	return round(number/max_s, 2)
def transform_to_state(robot_allies, robot_opponents, ball, enemy= False):
	state = []
	if(not enemy):
		state.append(scale(ball.body.position[0], 76))
		state.append(scale(ball.body.position[1], 60))
		state.append(scale(ball.body.linearVelocity[0], 50))
		state.append(scale(ball.body.linearVelocity[1], 50))
		state.append(scale(distance_between_ball_and_goal(ball.body),152))
		for a in range(NUMBER_OF_PLAYERS):
			state.append(scale(robot_allies[a].body.position[0],76))
			state.append(scale(robot_allies[a].body.position[1],60))
			state.append(scale(robot_allies[a].body.angle,2*math.pi))
			state.append(scale(robot_allies[a].body.linearVelocity[0],50))
			state.append(scale(robot_allies[a].body.linearVelocity[1],50))
			state.append(scale(robot_allies[a].body.angularVelocity, 6))
			state.append(scale(distance_between_bodies(robot_allies[a].body, ball.body), 152))
			state.append(scale(angle_between_bodies(robot_allies[a].body, ball.body),2*math.pi))

		for a in range(NUMBER_OF_ENEMIES):
			state.append(scale(robot_opponents[a].body.position[0],76))
			state.append(scale(robot_opponents[a].body.position[1],60))
			state.append(scale(robot_opponents[a].body.angle,2*math.pi))
			state.append(scale(robot_opponents[a].body.linearVelocity[0],50))
			state.append(scale(robot_opponents[a].body.linearVelocity[1],50))
			state.append(scale(robot_opponents[a].body.angularVelocity, 6))
			state.append(scale(distance_between_bodies(robot_opponents[a].body, ball.body), 152))
			state.append(scale(angle_between_bodies(robot_opponents[a].body, ball.body),2*math.pi))
		#print('distance between: ', distance_between_bodies(robot_allies[0].body, ball.body))
		#print('angle between: ', math.degrees(angle_between_bodies(robot_allies[a].body, ball.body)))
		state = np.array(state)
		state = state.reshape(NUM_FEATURES,1)
	else :
		state.append(scale(-ball.body.position[0], 76))
		state.append(scale(-ball.body.position[1], 60))
		state.append(scale(-ball.body.linearVelocity[0], 50))
		state.append(scale(-ball.body.linearVelocity[1], 50))
		state.append(scale(distance_between_ball_and_goal(ball.body),152))
		for a in range(NUMBER_OF_PLAYERS):
			state.append(scale(-robot_allies[a].body.position[0],76))
			state.append(scale(-robot_allies[a].body.position[1],60))
			state.append(scale(robot_allies[a].body.angle + math.pi,2*math.pi,angle=True))
			#state.append(scale(robot_allies[a].body.angle + math.pi,2*math.pi,angle=False))
			state.append(scale(robot_allies[a].body.linearVelocity[0],50))
			state.append(scale(robot_allies[a].body.linearVelocity[1],50))
			state.append(scale(robot_allies[a].body.angularVelocity, 6))
			state.append(scale(distance_between_bodies(robot_allies[a].body, ball.body), 152))
			#state.append(scale(angle_between_bodies(robot_allies[a].body, ball.body, angle=True),math.pi))
			state.append(scale(angle_between_bodies(robot_allies[a].body, ball.body, angle=False),math.pi))
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
	def __init__(self, enemy=False):
		self.isEnemy = enemy
		self.decrease = 0.0
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
		self.goaldone = False
		# for angle in range(MIN_ANG_SPEED, MAX_ANG_SPEED+ ANG_STEP, ANG_STEP):
		# 	for linear in range(MIN_LIN_SPEED, MAX_LIN_SPEED +LIN_STEP, LIN_STEP):
		# 		self.action_space.append((angle, linear))
		for angle in range(-MAX_ANG_ACCEL, MAX_ANG_ACCEL+ 1):
			for linear in range(-MAX_LIN_ACCEL, MAX_LIN_ACCEL +1):
				self.action_space.append((angle, linear))
		self.speed = [[0,0],[0,0],[0,0],[0,0],[0,0]]
		self.model_name = 'mymodel_episodic_v2.h5'
		if(self.isEnemy):
			self.model_name= 'enemy_' + self.model_name
		self.action = None
		if(only_play or load_model):
			self.model = nn.neural_net_model4(NUMBER_OF_PLAYERS, self.model_name)
		else :
			self.model = nn.neural_net_model4(NUMBER_OF_PLAYERS)
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
		self.player_memory.append((player.body.position[0], player.body.position[1]))
	#TODO
	def getReward(self, new_state, robot_allies, robot_opponents, ball):
		reward = 0 #- self.t_hits//30
		ball_x = ball.body.position[0]
		if(ball_x <= BALL_MIN_X and not self.goaldone):
			if(self.isEnemy):
				reward = self.reward['goal']
			else:
				reward = self.reward['enemy_goal']
			self.goaldone = True
			print('goal!!')
		elif(ball_x >= BALL_MAX_X and not self.goaldone):
			if(self.isEnemy):
				reward = self.reward['enemy_goal']
			else :
				reward = self.reward['goal']
			self.goaldone = True
			print('goall!!!!')
		elif(self.playerHitBall(robot_allies, robot_opponents, ball)):
			self.t_hits = 0
			reward = 200
		elif(self.times%MAX_FRAMES_GAME == 0 and ball_x < BALL_MAX_X):
			self.restart = True
			reward = -500


		self.t_hits+=1
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
		for allies in robot_allies:
			if(allies.body.userData == ball.body):
				print('hit!')
				allies.body.userData = None
				return True
		return False


	def compute(self, robot_allies, robot_opponents, ball):
		if(self.times%3 != 0 and not only_play):
			self.times+=1
			return
		new_state = transform_to_state(robot_allies, robot_opponents, ball)
		reward = self.getReward(new_state, robot_allies, robot_opponents, ball)
		if(len(self.replay_memory) < OBSERVE_TIMES-1 and (not self.restart)):
			self.replay_memory.append((self.old_state[:], self.action_number_ang, self.action_number_lin, reward, new_state[:]))
		else :
			self.replay_memory.append((self.old_state[:], self.action_number_ang, self.action_number_lin, reward, new_state[:]))
			batch = self.replay_memory
			self.replay_memory = []
			x_train, angv1_train, linv1_train, angv2_train, linv2_train, angv3_train, linv3_train = self.generate_train_from_batch(batch)
			print('fitting...')
			self.model.fit(x_train, [angv1_train, linv1_train, angv2_train, linv2_train, angv3_train, linv3_train], batch_size=BATCH_SIZE, epochs=10, verbose=0)
		self.add_ball_memory(ball)
		self.add_player_memory(robot_allies[0])
		if(self.times%MAX_FRAMES_GAME==0):
			print('saving and restarting...')
			self.episodes+=1
			self.restart = True
			self.model.save_weights(self.model_name)
		if(self.restart):
			self.treatRestart()
		self.times+=1
		if(self.episodes > MAX_EPISODES*10):#self.times > MAX_FRAMES*10):
			print('saving and exiting ...')
			self.model.save(self.model_name)
			exit()

	def treatRestart(self):
		self.goaldone = False
		self.replay_memory = []
		self.last_speeds = deque()
		self.ball_memory = deque()
		self.player_memory = deque()

	def isTerminalState(self,reward):
		if(reward == -500 or reward == -100 or reward == self.reward['goal'] or reward == self.reward['enemy_goal']):
			return True
		return False


	def generate_train_from_batch(self, batch):
		x_train = []
		angv1_train = []
		linv1_train = []
		angv2_train = []
		linv2_train = []
		angv3_train = []
		linv3_train = []
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
			else :
				update = reward
			#decay = max((ALPHA - self.times/(MAX_FRAMES*6)), 0.15)
			decay = ALPHA
			y_train = []
			for a in range(NUMBER_OF_PLAYERS):
				y = []
				y[:] = old_qval[0][:]
				old_qval_action = y[0][action_number_ang[a]]
				y[0][action_number_ang[a]] = (1-decay)*old_qval_action + decay*update

				y2 = []
				y2[:] = old_qval[1][:]
				old_qval_action = y2[0][action_number_lin[a]]
				y2[0][action_number_lin[a]] = (1-decay)*old_qval_action + decay*update
				y_train.append(y[:])
				y_train.append(y2[:])
			
			angv1 = np.array(y_train[0])
			linv1 = np.array(y_train[1])
			angv2 = np.array(y_train[2])
			linv2 = np.array(y_train[3])
			angv3 = np.array(y_train[4])
			linv3 = np.array(y_train[5])
			x_train.append(old_state.reshape((NUM_FEATURES),))

			angv1_train.append(angv1.reshape(NB_ANG_ACT))
			linv1_train.append(linv1.reshape(NB_LIN_ACT))
			angv2_train.append(angv2.reshape(NB_ANG_ACT))
			linv2_train.append(linv2.reshape(NB_LIN_ACT))
			angv3_train.append(angv3.reshape(NB_ANG_ACT))
			linv3_train.append(linv3.reshape(NB_LIN_ACT))

		x_train = np.array(x_train)
		angv1_train = np.array(angv1_train)
		linv1_train = np.array(linv1_train)
		angv2_train = np.array(angv2_train)
		linv2_train = np.array(linv2_train)
		angv3_train = np.array(angv3_train)
		linv3_train = np.array(linv3_train)

		return x_train, angv1_train, linv1_train, angv2_train, linv2_train,angv3_train, linv3_train


	def action_saturate(self, acts):
		speed = []
		for act in acts:
			angle_v, lin_v = act
			lin_v = min(max(MIN_LIN_SPEED, lin_v), MAX_LIN_SPEED)
			angle_v = min(max(MIN_ANG_SPEED, angle_v), MAX_ANG_SPEED)
			speed.append([angle_v, lin_v])
		return speed


	def sync_control_centrallized(self, ally_positions, enemy_positions, ball):
		if(only_play):
			self.times +=1
		if(self.times%3!=0 and not only_play):
			return
		state = transform_to_state(ally_positions, enemy_positions, ball) #enemy=self.isEnemy)
		self.old_state = state[:]
		#print('state',state.transpose())
		dec = max(EPSILON - self.decrease, 0.01)
		print('EPSILON', dec)
		action_ang = []
		action_lin = []
		if((random.random() < dec or self.times < OBSERVE_TIMES) and not only_play):
			for i in (range(NUMBER_OF_PLAYERS)):
				action_ang.append(random.randint(0, (len(ANG_ACCEL_VEC)-1)))
				action_lin.append(random.randint(0, (len(LIN_ACCEL_VEC)-1)))
			#action = (random.randint(0,NUMBER_OF_ACTIONS-1))
		else :
			predicted_qval = self.model.predict(state.transpose(), batch_size=1) #checar batch size!!
			for i in range(0,2*NUMBER_OF_PLAYERS, 2):
				action_ang.append(np.argmax(predicted_qval[i]))
				action_lin.append(np.argmax(predicted_qval[i+1]))
		i = 0
		for a,b in zip(action_ang, action_lin):
			self.speed[i][0] += ANG_ACCEL_VEC[a]
			self.speed[i][1] += LIN_ACCEL_VEC[b]
			i+=1
		# if(self.isEnemy):
		# 	ang_accel = -ang_accel
		# 	lin_accel = lin_accel
		self.action_number_ang = action_ang
		self.action_number_lin = action_lin
		self.speed = self.action_saturate(self.speed)[:]
		# print(self.speed)

		self.add_speed_memory(self.speed)
		allies = self.speed[:]
		#self.decrease = self.times/MAX_FRAMES#7000000
		self.decrease = self.episodes/(MAX_EPISODES*3)
		# print(self.times, 'out of ', MAX_FRAMES*10)
		print(self.episodes, 'out of ', MAX_EPISODES*10)
		return (allies)


def main():
	sc = SimController()
	while(True):
		sc.sync_update()


if __name__ == '__main__':
	main()