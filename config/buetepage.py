class global_config:
	def __init__(self):
		self.NUM_JOINTS = 4
		self.JOINTS_DIM = 3
		self.WINDOW_LEN = 1
		self.ROBOT_JOINTS = 7
		self.NUM_ACTIONS = 5
		self.optimizer = 'Adam'
		self.lr = 1e-3
		self.EPOCHS = 100
		self.EPOCHS_TO_SAVE = 5

class ae_config:
	def __init__(self):
		config = global_config()
		self.batch_size = 32
		self.num_joints = config.NUM_JOINTS
		self.joint_dims = config.JOINTS_DIM
		self.window_size = config.WINDOW_LEN
		self.hidden_sizes = [250, 150]
		self.latent_dim = 10
		self.beta = 0.5
		self.activation = 'ReLU'
		self.z_prior_mean = 0
		self.z_prior_std = 1
