class ae_config:
	def __init__(self):
		self.num_joints = 4
		self.joint_dims = 3
		self.hidden_sizes = [250, 150]
		self.activation = 'LeakyReLU'

class robot_vae_config:
	def __init__(self):
		self.num_joints = 7
		self.joint_dims = 1
		self.hidden_sizes = [250, 150]
		self.activation = 'LeakyReLU'

