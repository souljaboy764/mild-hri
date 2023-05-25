class global_config:
	def __init__(self):
		self.NUM_JOINTS = 4
		self.JOINTS_DIM = 3
		self.WINDOW_LEN = 40
		self.ROBOT_JOINTS = 7
		self.NUM_ACTIONS = 4
		self.optimizer = 'AdamW'
		self.lr = 1e-4
		self.EPOCHS = 100
		self.EPOCHS_TO_SAVE = 5

class ae_config:
	def __init__(self):
		config = global_config()
		self.num_joints = config.NUM_JOINTS
		self.joint_dims = config.JOINTS_DIM
		self.window_size = config.WINDOW_LEN
		self.hidden_sizes = [250, 150]
		self.latent_dim = 7
		self.beta = 0.05
		self.activation = 'LeakyReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1
		self.mce_samples = 4

class robot_vae_config:
	def __init__(self):
		config = global_config()
		self.num_joints = config.ROBOT_JOINTS
		self.joint_dims = 1
		self.window_size = config.WINDOW_LEN
		self.hidden_sizes = [250, 150]
		self.latent_dim = 7
		self.beta = 0.001
		self.activation = 'LeakyReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1
