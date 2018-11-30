#!/usr/bin/env python

import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')

import cPickle as pickle

import os.path

import time

import tensorflow as tf
import numpy as np
import rospy
import rospkg

from datetime import datetime

from environments.base_environment import Environment
from utils.memory import ExperienceBuffer
from algo.cpo import Actor, Critic, SafetyBaseline
from utils.common import *
from utils.map import *
from options import Options

args = Options().parse()

tf.reset_default_graph()

sess = tf.Session()

base_path = rospkg.RosPack().get_path("reinforcement_learning_navigation")
summary_filename = args.output_name

# Set up folder structure
date_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
storage_path = os.path.join(base_path, "logs", date_str + "_" + summary_filename)
weights_path = os.path.join(storage_path, "weights")
tensorboard_path = os.path.join(storage_path, "tensorboard")
diagnostics_path = os.path.join(storage_path, "diagnostics")
os.mkdir(storage_path)
os.mkdir(weights_path)
os.mkdir(tensorboard_path)
os.mkdir(diagnostics_path)

rew_disc_factor = args.rew_disc_factor
saf_disc_factor = args.saf_disc_factor
lamda = args.lamda
safety_lamda = args.safety_lamda
safety_desired_threshold = args.safety_desired_threshold
center_advantages = args.center_advantages
use_safety_baseline = args.use_safety_baseline

experience_batch_size = args.timesteps_per_epoch
n_epochs = args.n_epochs

map_size = args.map_size
map_resolution = args.map_resolution
map_strategy = args.map_strategy
obstacles_map = args.obstacles_map
obstacle_padding = args.obstacle_padding
free_padding = args.free_padding

obstacle_positions = get_obstacle_positions(map_size, obstacles_map)
obstacles_map = get_obstacles_map(map_size, obstacle_positions, map_resolution, obstacle_padding)
free_map = get_obstacles_map(map_size, obstacle_positions, map_resolution, free_padding)

arch = args.architecture
if(arch == 'asl'):
    n_states = 38
elif(arch == 'tai'):
    n_states = 12

action_dim = 2 #Translational and rotational velocity
trans_vel_limits = [args.trans_vel_low, args.trans_vel_high]
rot_vel_limits = [args.rot_vel_low, args.rot_vel_high]
std_trans_init = args.std_trans_init
std_rot_init = args.std_rot_init

environment = Environment(args, n_states)
environment.set_obstacles_map(obstacles_map, map_resolution)

actor_filename = None

if args.jump_start:
  print("Jump starting the model.")
  actor_filename = os.path.join(rospkg.RosPack().get_path("reinforcement_learning_navigation"), args.model_init)

print("Training setup:")
print("Initializing the model from: {}".format(actor_filename))
print("Translational velocity limits: {}".format(trans_vel_limits))
print("Rotational velocity limits: {}".format(rot_vel_limits))
print("The universal output file name is: {}".format(args.output_name))
print("Timesteps per epoch: {}".format(experience_batch_size))
print("Number of epochs: {}".format(n_epochs))

actor_desired_kl = args.actor_desired_kl
critic_desired_kl = args.critic_desired_kl
safety_baseline_desired_kl = args.safety_baseline_desired_kl

policy_estimator = Actor(n_states, action_dim, [trans_vel_limits, rot_vel_limits],
                                       [np.log(std_trans_init), np.log(std_rot_init)], actor_desired_kl, sess, arch, actor_filename)

value_estimator = Critic(n_states, critic_desired_kl, sess, arch, filename=None)
if(use_safety_baseline == True):
    value_estimator_safety = SafetyBaseline(n_states, safety_baseline_desired_kl, sess, arch, filename=None)
experience_buffer = ExperienceBuffer()

total_experiences = 0
episode_number = 0
episodes_this_epoch = 0

summary_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
summary_dict = {'episodes_in_epoch':np.array(()), 'average_reward_sum_per_episode':np.array(()), 'average_returns':np.array(()), 'average_reward_per_experience':np.array(()), 'average_experience_per_episode':np.array(()), 'average_safety_constraint':np.array(()), 'success_rate':np.array(()), 'crash_rate':np.array(())}
tf_ep_epoch = tf.placeholder(shape=[], dtype = tf.float32)
tf_ep_epoch_avg_rew = tf.placeholder(shape=[], dtype = tf.float32)
tf_ep_epoch_avg_disc_rew = tf.placeholder(shape=[], dtype = tf.float32)
tf_ex_epoch_avg_rew = tf.placeholder(shape = [], dtype = tf.float32)
tf_ex_per_ep_epoch_avg = tf.placeholder(shape = [], dtype = tf.float32)
tf_ep_epoch_avg_disc_safe_cost = tf.placeholder(shape = [], dtype = tf.float32)
tf_frac_goal_reached = tf.placeholder(shape = [], dtype = tf.float32)
tf_frac_crashed = tf.placeholder(shape = [], dtype = tf.float32)

with tf.variable_scope("Diagnostics"):
    ep_epoch_summary = tf.summary.scalar('episodes_in_epoch', tf_ep_epoch)
    ep_epoch_avg_rew_summary = tf.summary.scalar('average_reward_sum_per_episode', tf_ep_epoch_avg_rew)
    ep_epoch_avg_disc_rew_summary = tf.summary.scalar('average_discounted_reward_sum_per_episode', tf_ep_epoch_avg_disc_rew)
    ex_epoch_avg_rew_summary = tf.summary.scalar('average_reward_per_experience', tf_ex_epoch_avg_rew)
    ex_per_ep_epoch_avg_summary = tf.summary.scalar('average_experience_per_episode', tf_ex_per_ep_epoch_avg)
    ep_epoch_avg_disc_safe_cost_summary = tf.summary.scalar('average_discounted_safety_cost', tf_ep_epoch_avg_disc_safe_cost)
    tf_frac_goal_reached_summary = tf.summary.scalar('fraction_goal_reached', tf_frac_goal_reached)
    tf_frac_crashed_summary = tf.summary.scalar('fraction_crashed', tf_frac_crashed)
    summary_op = tf.summary.merge([ep_epoch_summary, ep_epoch_avg_rew_summary, ep_epoch_avg_disc_rew_summary, ex_epoch_avg_rew_summary, ex_per_ep_epoch_avg_summary, ep_epoch_avg_disc_safe_cost_summary, tf_frac_goal_reached_summary, tf_frac_crashed_summary])

sess.run(tf.global_variables_initializer())

graph = tf.get_default_graph()
graph.finalize()

epoch = 1
discounted_sum_rewards_counter = 0
discounted_sum_safety_costs_counter = 0
goal_reach_counter = 0
crash_counter = 0
safety_constraint_counter = 0
epoch_start_time = time.time()

print("Starting training.")

while(epoch <= n_epochs):

    map_choice = get_map_choice(map_strategy)

    robot_position = get_free_position(free_map, map_resolution, map_size/2, map_choice)
    robot_orientation = (2*np.random.rand() - 1)*np.pi
    environment.set_robot_pose([robot_position[0],robot_position[1],0.025], [0,0,robot_orientation])

    goal_position = get_free_position(free_map, map_resolution, map_size/2, map_choice)
    goal_orientation = (2*np.random.rand() - 1)*np.pi

    distance_map = get_distance_map(map_size, obstacles_map, goal_position, map_resolution)
    environment.set_distance_map(distance_map)
    goal = [[goal_position[0],goal_position[1],0], [0,0,goal_orientation]]
    environment.set_goal(goal)

    state = environment.reset()

    episode_number += 1
    episodes_this_epoch += 1

    experience_counter = 0
    episode_start_time = time.time()

    while(environment.is_running):
        #Run episode
        action = policy_estimator.predict_action(np.reshape(state, (-1, n_states)))
        next_state, reward, safety_cost, simulator_flag, _ = environment.execute_action(action)

        if(simulator_flag == True): #Workaround for stage simulator crashing and restarting
            episode_number -= 1
            episodes_this_epoch -= 1
            break

        experience_counter += 1
        experience_buffer.add_experience(state, action, reward, safety_cost, next_state)
        state = next_state

    if(simulator_flag == False):
        #Compute metrics
        goal_reach_counter += environment.goal_reached
        crash_counter += environment.crashed
        total_experiences += experience_counter
        experience_buffer.add_episode(rew_disc_factor, saf_disc_factor)
        states, _, rewards, safety_costs, next_states, discounted_rewards, discounted_safety_costs = experience_buffer.get_episode_experiences()
        safety_constraint_counter += discounted_safety_costs[0]

        discounted_sum_rewards_counter += discounted_rewards[0]
        discounted_sum_safety_costs_counter += discounted_safety_costs[0]

        #Compute advantages
        baseline_values = value_estimator.predict_value(states)
        future_discounted_state_estimates = rew_disc_factor*value_estimator.predict_value(next_states)
        td_errors = rewards + future_discounted_state_estimates - baseline_values
        generalized_advantages = get_generalized_advantages(np.reshape(td_errors,-1), rew_disc_factor, lamda, np.reshape(future_discounted_state_estimates,-1)[-1])

        #Compute safety advantages
        if(use_safety_baseline == True):
            safety_baseline_values = value_estimator_safety.predict_value(states)
            future_discounted_state_safety_estimates = saf_disc_factor*value_estimator_safety.predict_value(next_states)
            td_errors_safety = safety_costs + future_discounted_state_safety_estimates - safety_baseline_values
            generalized_advantages_safety = get_generalized_advantages(np.reshape(td_errors_safety,-1), saf_disc_factor, safety_lamda, np.reshape(future_discounted_state_safety_estimates,-1)[-1])
        else:
            generalized_advantages_safety = get_generalized_advantages(np.reshape(safety_costs,-1), saf_disc_factor, safety_lamda, 0)

        experience_buffer.add_advantages(generalized_advantages, generalized_advantages_safety)

        if (total_experiences/(experience_batch_size*epoch) >= 1):

            safety_constraint = safety_desired_threshold - float(safety_constraint_counter)/episodes_this_epoch

            print('epoch {}, Updating after {} episodes, Time for epoch {}'.format(epoch, episodes_this_epoch, (time.time() - epoch_start_time)))
            print('Safety Constraint: ', safety_constraint)

            states, actions, rewards, safety_costs, _, discounted_future_rewards, discounted_future_safety_costs = experience_buffer.get_experiences()
            advantages, safety_advantages = experience_buffer.get_advantages()

            if(center_advantages):
                advantages = advantages - advantages.mean()
                safety_advantages = safety_advantages - safety_advantages.mean()

            old_action_means, old_action_stddevs = policy_estimator.update_weights(states, actions, advantages, safety_advantages, safety_constraint, episodes_this_epoch)
            old_predicted_targets = value_estimator.update_weights(states, discounted_future_rewards)

            crash_rate = float(crash_counter)/episodes_this_epoch
            success_rate = float(goal_reach_counter)/episodes_this_epoch
            print('Crash Rate: ', crash_rate)
            print('Success Rate: ', success_rate)

            summary = sess.run(summary_op, feed_dict = {tf_ep_epoch: episodes_this_epoch, tf_ex_epoch_avg_rew: np.mean(rewards), tf_ep_epoch_avg_rew: float(np.sum(rewards))/episodes_this_epoch, tf_ep_epoch_avg_disc_rew: float(discounted_sum_rewards_counter)/episodes_this_epoch, tf_ex_per_ep_epoch_avg: len(rewards)/episodes_this_epoch, tf_ep_epoch_avg_disc_safe_cost: float(discounted_sum_safety_costs_counter)/episodes_this_epoch, tf_frac_goal_reached: float(goal_reach_counter)/episodes_this_epoch, tf_frac_crashed: float(crash_counter)/episodes_this_epoch})

            summary_dict['episodes_in_epoch'] = np.append(summary_dict['episodes_in_epoch'],episodes_this_epoch)
            summary_dict['average_reward_sum_per_episode'] = np.append(summary_dict['average_reward_sum_per_episode'],float(np.sum(rewards))/episodes_this_epoch)
            summary_dict['average_returns'] = np.append(summary_dict['average_returns'],float(discounted_sum_rewards_counter)/episodes_this_epoch)
            summary_dict['average_reward_per_experience'] = np.append(summary_dict['average_reward_per_experience'],np.mean(rewards))
            summary_dict['average_experience_per_episode'] = np.append(summary_dict['average_experience_per_episode'],len(rewards)/episodes_this_epoch)
            summary_dict['average_safety_constraint'] = np.append(summary_dict['average_safety_constraint'],float(discounted_sum_safety_costs_counter)/episodes_this_epoch)
            summary_dict['success_rate'] = np.append(summary_dict['success_rate'],success_rate)
            summary_dict['crash_rate'] = np.append(summary_dict['crash_rate'],crash_rate)

            summary1, kl = policy_estimator.get_summary(states, actions, advantages, safety_advantages, episodes_this_epoch, old_action_means, old_action_stddevs)
            summary2 = value_estimator.get_summary(states, discounted_future_rewards, old_predicted_targets)
            summary_writer.add_summary(summary, epoch)
            summary_writer.add_summary(summary1, epoch)
            summary_writer.add_summary(summary2, epoch)

            if(use_safety_baseline == True):
                old_predicted_targets_safety = value_estimator_safety.update_weights(states, discounted_future_safety_costs)
                summary3 = value_estimator_safety.get_summary(states, discounted_future_safety_costs, old_predicted_targets_safety)
                summary_writer.add_summary(summary3, epoch)

            pickle.dump(summary_dict, open(os.path.join(diagnostics_path, summary_filename+'.p'), 'wb'))

            summary_writer.flush()

            if(epoch % args.save_weights_freq == 0):
                pickle.dump(policy_estimator.get_weights(), open(os.path.join(weights_path, "weights_actor"+str(epoch)+'.p'), 'wb'))
                pickle.dump(value_estimator.get_weights(), open(os.path.join(weights_path, "weights_critic"+str(epoch)+'.p'), 'wb'))

            epoch += 1
            epoch_start_time = time.time()
            episodes_this_epoch = 0
            discounted_sum_rewards_counter = 0
            discounted_sum_safety_costs_counter = 0
            goal_reach_counter = 0
            crash_counter = 0
            safety_constraint_counter = 0
            experience_buffer.clear_buffer()

    experience_buffer.clear_episode_buffer()


summary_writer.close()

print("Training Finished")
