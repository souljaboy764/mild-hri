class global_config:
	def __init__(self):
		self.num_joints = 4
		self.joint_dims = 3
		self.window_size = 5
		self.downsample = 0.2
		self.robot_joints = 7
		self.num_actions = 4
		self.optimizer = 'AdamW'
		self.lr = 1e-4
		self.EPOCHS = 100
		self.EPOCHS_TO_SAVE = 5
		self.beta = 0.005

class ae_config:
	def __init__(self):
		config = global_config()
		self.num_joints = config.num_joints
		self.joint_dims = config.joint_dims
		self.window_size = config.window_size
		self.hidden_sizes = [40, 20]
		self.latent_dim = 7
		self.beta = config.beta
		self.activation = 'LeakyReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1
		self.mce_samples = 10

class robot_vae_config:
	def __init__(self):
		config = global_config()
		self.num_joints = config.robot_joints
		self.joint_dims = 1
		self.window_size = config.window_size
		self.hidden_sizes = [40, 20]
		self.latent_dim = 7
		self.beta = config.beta
		self.activation = 'LeakyReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1
