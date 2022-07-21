#!/usr/bin/python
# -*- coding:utf-8 -*-

# -----------------------------------
# Pepper Control with Moveit for MILD-HRI
# Adapted from https://github.com/souljaboy764/icra_handshaking/blob/cec1e7962e57a7e1cb965a61add4e20563ce6379/src/pepper_promp_moveit.py and https://github.com/souljaboy764/icra_handshaking/blob/master/src/nuitrack_skeleton_predictor.py
# Author: souljaboy764
# Date: 2021/5/23
# -----------------------------------


import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os

from utils import joints_dic, joints_in, joint_angle_extraction, rotation_normalization
from networks import FullCovVAE

# ROS
from std_msgs.msg import Float32MultiArray

import rospy
import rospkg

import matplotlib.pyplot as plt

class MILDHRIROS:
	def __init__(self, input_dim=15*3, hidden_dim=64, checkpoint='logs/fullvae_nui_07150151_alphaargmaxnocond_pretrained/models/final.pth'):
		print('Creating Model')
		hyperparams = np.load(os.path.join(os.path.dirname(checkpoint),'hyperparams.npz'), allow_pickle=True)
		saved_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
		vae_config = hyperparams['ae_config'].item()
		vae_config.window_size = 5
		self._model = FullCovVAE(**(vae_config.__dict__)).to(device)
		self.z_dim = self._model.latent_dim
		print("Loading Checkpoints")
		try:
			checkpoint = torch.load(checkpoint,device)
			self._model.load_state_dict(checkpoint['model'])
			self._hsmm = checkpoint['hsmm']
		except Exception as e:
			print('Exception occurred while loading checkpoint:')
			print(e)
			exit(-1)
		self.skeleton_trajectory = []
		self._model.eval()

		# self.joint_idx_list = np.array([joints_dic["neck"], joints_dic["right_shoulder"], joints_dic["right_elbow"], joints_dic["right_wrist"]])

		# Transform from base_link to camera
		self.first_msg = None
		# actions = ['clapfist2', 'fistbump2', 'handshake2', 'highfive1', 'rocket1', 'wave1']
		self.action = 1
		self.robot_command = None

		self._nui_sub = rospy.Subscriber("/perception/skeletondata_0", Float32MultiArray, self.skeletonCb)
		rospy.loginfo('MILDHRIROS Ready!')

	def skeletonCb(self, msg):
		# msg.data = np.array(msg.data)
		# skeleton = msg.data.reshape(-1,3)
		# print(skeleton.shape)
		
		# # Compute matrix for pose normalization
		# if len(self.skeleton_trajectory) == 0: 
		# 	self._rotMat = rotation_normalization(skeleton)
		
		# skeleton = self._rotMat[:3,:3].dot(skeleton.T) + np.expand_dims(self._rotMat[:3,3],-1)
		
		# self.skeleton_trajectory.append(np.array([skeleton[self.joint_idx_list]]).flatten())
		# self.skeleton_trajectory.append(np.array(msg.data))

		skeleton = np.array(msg.data).reshape(-1,3)
		if len(self.skeleton_trajectory)==0:
			self._rotMat = np.eye(4)
			self._rotMat[:3,:3] = rotation_normalization(skeleton)
			self._rotMat[:3,3] = skeleton[0]
		skeleton -= self._rotMat[:3,3]
		skeleton = self._rotMat[:3,:3].dot(skeleton.T).T
		skeleton += np.array([-2.4, 1.3, -0.5]) # Needed for moving it to the general space of NuiSI trajectories
		print(skeleton.shape)
		joints_idx = np.array([joints_dic[i] for i in joints_in])
		self.skeleton_trajectory.append(skeleton[joints_idx])#.flatten())
		# skeleton = np.concatenate([-skeleton[:, 2:3], skeleton[:, 0:1], -skeleton[:, 1:2]],axis=-1)
		# R = np.array([[ 0.70710678, -0.35355339,  0.61237244],
		# 				[ 0.        ,  0.8660254 ,  0.5       ],
		# 				[-0.70710678, -0.35355339,  0.61237244]])
		# skeleton = R.dot(skeleton.T).T
		# skeleton += np.array([[-1.5,1.6,-1.2]])
		# self.skeleton_trajectory.append(skeleton.flatten())


		if len(self.skeleton_trajectory)<self._model.window_size+3:
			return
		# if len(self.skeleton_trajectory)>self._model.window_size:
		# 	self.skeleton_trajectory.pop(0)
		idx = np.array([np.arange(n,n+self._model.window_size) for n in range(len(self.skeleton_trajectory) + 1 - self._model.window_size)])
		
		# Forward pass based on the observed skeleton trajectory till now
		with torch.no_grad():
			print(np.array(self.skeleton_trajectory).shape, np.array(self.skeleton_trajectory)[idx].shape, idx.shape[0])
			xh = torch.Tensor(np.array(self.skeleton_trajectory)[idx].reshape((idx.shape[0],self._model.input_dim))).to(device)
			# print(xh.shape)
			zh = self._model(xh, encode_only=True)
			if np.isnan(zh.cpu().numpy()).any():
				print('isnan zh!',rospy.Time.now())
			print('zh.shape',zh.shape)
			self.robot_command = zh.cpu().numpy()
			alpha_hsmm, _, _, _, _ = self._hsmm[self.action].compute_messages(demo=zh.cpu().numpy(),marginal=slice(0, self.z_dim))
			if np.isnan(alpha_hsmm).any():
				print('isnan alpha_hsmm!',rospy.Time.now())
			zr, _ = self._hsmm[self.action].condition(zh.cpu().numpy(), dim_in=slice(0, self.z_dim), dim_out=slice(self.z_dim, 2*self.z_dim))
			if np.isnan(zr).any():
				print('isnan zr!',rospy.Time.now())
				# return
			print('zr.shape',zr.shape)
			xr_gen = self._model._output(self._model._decoder(torch.Tensor(zr[None,-1]).to(device)))
			xr_gen = xr_gen[0,-self._model.input_dim:] # taking the last prediction to send to pepper
			xr_gen = xr_gen.reshape((self._model.window_size,self._model.num_joints,3)).cpu().numpy()
		
		# self.robot_command = joint_angle_extraction(xr_gen[-1])

		# #TODO: Do some filtering of the poses before sending to pepper

		if len(self.skeleton_trajectory)>100:
			rospy.signal_shutdown('done')

if __name__=='__main__':
	rospy.init_node('mild_hri_node')

	rate = rospy.Rate(30)

	limits_max = [2.08567, -0.00872665, 2.08567, 1.56207]
	limits_min = [-2.08567, -1.56207, -2.08567, 0.00872665]
	bounds = ((limits_min[0], limits_max[0]),(limits_min[1], limits_max[1]),(limits_min[2], limits_max[2]),(limits_min[3], limits_max[3]))

	# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

	# robot_traj_publisher = rospy.Publisher("/pepper_dcm/RightArm_controller/command", JointTrajectory, queue_size=1)
	# joint_trajectory = JointTrajectory()
	# joint_trajectory.header.frame_id = "base_footprint"
	# joint_trajectory.joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
	# joint_trajectory.points.append(JointTrajectoryPoint())
	# joint_trajectory.points[0].time_from_start = rospy.Duration.from_sec(0.02)

	# from moveit_msgs.msg import DisplayRobotState
	# robotstate_pub = rospy.Publisher("display_robot_state", DisplayRobotState, queue_size=5)
	# robotmsg = DisplayRobotState()
	# robotmsg.state.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
	joints_msg = Float32MultiArray()
	joint_array_pub = rospy.Publisher("mild_hri_command", Float32MultiArray, queue_size=5)
	predictor = MILDHRIROS()

	while predictor.robot_command is None and not rospy.is_shutdown():
		rate.sleep()

	if rospy.is_shutdown():
		exit(-1)

	z_bag = []
	while not rospy.is_shutdown():
		rate.sleep()
		# last_q = np.clip(predictor.robot_command, limits_min, limits_max)
		# joint_values = np.hstack([last_q, [0.]])
		# joint_values = np.array([1.28193268, -0.2122131, 1.12623912, 1.07140932])

		# joints_msg.data = np.clip(predictor.robot_command, limits_min, limits_max)
		# joints_msg.data = predictor.robot_command
		# joint_array_pub.publish(joints_msg)
		if len(z_bag) > 0 and z_bag[-1] == predictor.robot_command:
			continue
		
		z_bag.append(predictor.robot_command)
		if len(z_bag)>80:
			break
		# joint_trajectory.points[0].positions = joint_values
		# joint_trajectory.header.stamp = rospy.Time.now()
		# robot_traj_publisher.publish(joint_trajectory)

		# robotmsg.state.joint_state.position = joint_values
		# # print(last_q, robotmsg.state.joint_state.position)
		# robotstate_pub.publish(robotmsg)

	z_bag = np.array(z_bag)
	data = np.load('logs/nuisi_results/predictions.npz',allow_pickle=True)
	z_test = data['z_data'][::2,:5]

	fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))

	ax.scatter(z_bag[:,0], z_bag[:,1], z_bag[:,2], color='r', marker='o', s=50, alpha=0.15)
	ax.scatter(z_test[:,0], z_test[:,1], z_test[:,2], color='b', marker='+', s=50, alpha=0.15)
	plt.show()


