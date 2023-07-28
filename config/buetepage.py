class ae_config:
	def __init__(self):
		self.num_joints = 3
		self.joint_dims = 3
		self.hidden_sizes = [40, 40]
		self.latent_dim = 7
		self.activation = 'LeakyReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1
		self.mce_samples = 4

class robot_vae_config:
	def __init__(self):
		self.num_joints = 7
		self.joint_dims = 1
		self.hidden_sizes = [20, 20]
		self.latent_dim = 7
		self.activation = 'LeakyReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1
		self.mce_samples = 4
