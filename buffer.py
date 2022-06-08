import numpy as np
import random
from collections import deque


class MemoryBuffer:

	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0

	def sample(self, batch_size):
		"""
		Samples a random batch from the replay memory buffer
		Args:
			count: batch size
		Return: 
			batch (numpy array): state_array, action_array, reward_array, next_state_array
		"""
		batch = []
		batch_size = min(batch_size, self.len)
		batch = random.sample(self.buffer, batch_size)

		s_arr = np.float32([arr[0] for arr in batch])
		a_arr = np.float32([arr[1] for arr in batch])
		r_arr = np.float32([arr[2] for arr in batch])
		s1_arr = np.float32([arr[3] for arr in batch])

		return s_arr, a_arr, r_arr, s1_arr

	def len(self):
		return self.len

	def add(self, s, a, r, s1):
		"""
		Adds a particular transaction in the memory buffer
		Args:
			s: current state
			a: action taken
			r: reward received
			s1: next state
		Return:
			None
		"""
		transition = (s,a,r,s1)
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append(transition)
