class ae_config:
	def __init__(self):
		self.num_joints = 3
		self.joint_dims = 3
		self.hidden_sizes = [40, 20]
		self.activation = 'LeakyReLU'

class robot_vae_config:
	def __init__(self):
		self.num_joints = 7
		self.joint_dims = 1
		self.hidden_sizes = [40, 20]
		self.activation = 'LeakyReLU'

