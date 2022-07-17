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

from utils import *
from networks import FullCovVAE

# ROS
from std_msgs.msg import Float32MultiArray

import rospy
import rospkg

class MILDHRIROS:
	def __init__(self, input_dim=15*3, hidden_dim=64, checkpoint=os.path.join(rospkg.RosPack().get_path('icra_handshaking'),'final.pth')):#, promp_data, ):
		print('Creating Model')
		hyperparams = np.load(os.path.join(os.path.dirname(checkpoint),'hyperparams.npz'), allow_pickle=True)
		saved_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
		vae_config = hyperparams['ae_config'].item()
		self._model = FullCovVAE(**(vae_config.__dict__)).to(device)
		self.z_dim = self._model.latent_dim
		print("Loading Checkpoints")
		try:
			checkpoint = torch.load(checkpoint)
			self._model.load_state_dict(checkpoint['model'])
			self._hsmm = checkpoint['hsmm']
		except Exception as e:
			print('Exception occurred while loading checkpoint:')
			print(e)
			exit(-1)
		self.skeleton_trajectory = []
		self._model.eval()

		self.joint_idx_list = np.array([joints_dic["neck"], joints_dic["right_shoulder"], joints_dic["right_elbow"], joints_dic["right_wrist"]])

		# Transform from base_link to camera
		self.first_msg = None
		# actions = ['clapfist2', 'fistbump2', 'handshake2', 'highfive1', 'rocket1', 'wave1']
		self.action = 4
		self.robot_command = None

		self._nui_sub = rospy.Subscriber("/perception/skeletondata_0", Float32MultiArray, self.skeletonCb)
		rospy.loginfo('MILDHRIROS Ready!')

	def skeletonCb(self, msg):
		msg.data = np.array(msg.data)
		skeleton = msg.data.reshape(-1,3)
		
		# Compute matrix for pose normalization
		if len(self.skeleton_trajectory) == 0: 
			self._rotMat = rotation_normalization(skeleton)
		
		skeleton = self._rotMat[:3,:3].dot(skeleton.T) + np.expand_dims(self._rotMat[:3,3],-1)
		
		self.skeleton_trajectory.append(np.array([skeleton[self.joint_idx_list]]).flatten())
		if len(self.skeleton_trajectory)<self._model.window_size:
			return
		# if len(self.skeleton_trajectory)>self._model.window_size:
		# 	self.skeleton_trajectory.pop(0)
		idx = np.array([np.arange(n,n+self._model.window_size) for n in range(len(self.skeleton_trajectory) + 1 - self._model.window_size)])
		
		# Forward pass based on the observed skeleton trajectory till now
		xh = torch.Tensor(np.vstack(self.skeleton_trajectory)[idx].reshape((1,idx.shape[0],self._model.input_dim))).to(device)
		zh = self._model(xh, encode_only=True)
		zr, _ = self.hsmm[self.action].condition(zh.detach().cpu().numpy(), dim_in=slice(0, self.z_dim), dim_out=slice(self.z_dim, 2*self.z_dim))
		xr_gen = self._model._output(self._model._decoder(torch.Tensor(zr).to(device)))[0,-1,-self._model.input_dim:] # taking the last prediction to send to pepper
		xr_gen = xr_gen.reshape((-1,3))

		self.robot_command = joint_angle_extraction(xr_gen)

		#TODO: Do some filtering of the poses before sending to pepper

		if len(self.skeleton_trajectory)>70:
			rospy.signal_shutdown('done')

if __name__=='__main__':
	rospy.init_node('mild_hri_moveit_node')

	rate = rospy.Rate(30)

	limits_max = [2.08567, -0.00872665, 2.08567, 1.56207]
	limits_min = [-2.08567, -1.56207, -2.08567, 0.00872665]
	bounds = ((limits_min[0], limits_max[0]),(limits_min[1], limits_max[1]),(limits_min[2], limits_max[2]),(limits_min[3], limits_max[3]))

	from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

	robot_traj_publisher = rospy.Publisher("/pepper_dcm/RightArm_controller/command", JointTrajectory, queue_size=1)
	joint_trajectory = JointTrajectory()
	joint_trajectory.header.frame_id = "base_link"
	joint_trajectory.joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
	joint_trajectory.points.append(JointTrajectoryPoint())
	joint_trajectory.points[0].time_from_start = rospy.Duration.from_sec(0.02)

	predictor = MILDHRIROS()

	while predictor.robot_command is None and not rospy.is_shutdown():
		rate.sleep()

	if rospy.is_shutdown():
		exit(-1)

	while not rospy.is_shutdown():
		rate.sleep()
		last_q = np.clip(predictor.robot_command, limits_min, limits_max)
		joint_values = np.hstack([last_q, [np.pi/2]])
		joint_trajectory.points[0].positions = joint_values
		joint_trajectory.header.stamp = rospy.Time.now()
		robot_traj_publisher.publish(joint_trajectory)
