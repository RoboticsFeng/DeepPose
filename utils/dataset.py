import sys
sys.path.append('..')

import numpy as np
import random
import pickle
import net.config as cfg


class dataset(object):

	def __init__(self, data_size = 100):
		self.path = cfg.PATH + '/data/train/all/'
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.view_points = cfg.VIEW_POINTS
		self.batch_size = cfg.BATCH_SIZE

		self.indice_list = list(range(data_size))
		random.shuffle(self.indice_list)
		self.start_indices = 0
		self.data_step = 0

	def get_next_batch(self):
		image_batch = np.empty((self.batch_size, self.image_size, self.image_size, 3), dtype = 'float32')
		label_batch = np.empty((self.batch_size, self.cell_size, self.cell_size, 4 + self.view_points), dtype = 'float32')
		for i in range(self.batch_size):
			indice = i + self.start_indices
			training_data_name = self.path + str(indice).zfill(5)
			training_data = pickle.load(open(training_data_name, 'rb'))

			image_batch[i, :, :, :] = training_data[0]
			label_batch[i, :, :, :] = training_data[1]

		self.start_indices = self.start_indices + self.batch_size
		self.data_step = self.data_step + 1

		if self.start_indices + self.batch_size > len(self.indice_list):
		# update the args after all training data is fed
			random.shuffle(self.indice_list)
			self.start_indices = 0
		return image_batch, label_batch

