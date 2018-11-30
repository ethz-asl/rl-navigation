import argparse

class Options():

    def __init__(self):
        parser = argparse.ArgumentParser(description='RL training for navigation using constrained policy optimization')

        parser.add_argument('--output_name', help='Name used as base for all the output files.', type=str, default='tmp_model')
        parser.add_argument('--jump_start', help='If the training should be jump-started from IL', type=int, default=0)
        parser.add_argument('--model_init', help='Model used for initialization of the weights.', type=str)
        parser.add_argument('--timesteps_per_epoch', help='Number of timesteps per epoch weight update.', type=int, default=60000)
        parser.add_argument('--n_epochs', help='Number of total training epochs.', type=int, default=1000)
        parser.add_argument('--save_weights_freq', help='Frequency of saving weights.', type=int, default=100)

        parser.add_argument('--architecture', help='Architecture to use for policy model (asl or tai). asl uses 36 laser inputs and tanh activations, whereas tai uses 10 laser inputs and relu activations', type=str, default='asl')
        parser.add_argument('--use_safety_cost', type=bool, default=True, help='If true, uses a safety cost for the unsafe states. If not, uses a fixed negative reward for the unsafe states.')

        parser.add_argument('--rew_disc_factor', type=float, default=0.99, help='Discount factor to use for the rewards.')
        parser.add_argument('--saf_disc_factor', type=float, default=0.99, help='Discount factor to use for the safety costs.')
        parser.add_argument('--lamda', type=float, default=0.96, help='Lambda to use for generalized advantage estimation of the rewards.')
        parser.add_argument('--safety_lamda', type=float, default=0.75, help='Lambda to use for generalized advantage estimation of the safety costs.')
        parser.add_argument('--safety_desired_threshold', type=float, default=0.4, help='Threshold to use for average discounted safety returns.')
        parser.add_argument('--center_advantages', type=bool, default=True, help='If true, mean is subtracted after computing the advantages.')
        parser.add_argument('--use_safety_baseline', type=bool, default=False, help='If true, use an additional estimator to predict the safety value function(similar to value function estimation).')
        parser.add_argument('--actor_desired_kl', type=float, default=0.002, help='KL divergence for actor update.')
        parser.add_argument('--critic_desired_kl', type=float, default=0.002, help='KL divergence for critic update.')
        parser.add_argument('--safety_baseline_desired_kl', type=float, default=0.002, help='KL divergence for safety baseline update.')

        parser.add_argument('--map_size', type=float, default=10.0, help='Uses a square map of the size specified. Size needs to be the same as one used in stage.')
        parser.add_argument('--map_resolution', type=float, default=0.1, help='Resolution of the map to be used. Used in computing the free grid-points in the map and shortest path disctance.')
        parser.add_argument('--map_strategy', type=str, default='random-sampling', help='Choice of the map for the navigation task, \'random-sampling\' to choose a map randomly. Otherwise use \'map-1\', \'map-2\' etc to use a specific map.')
        parser.add_argument('--obstacles_map', type=str, default='train_map', help='Name of the numpy file which contains the positions of obstacles.')
        parser.add_argument('--obstacle_padding', type=float, default=0.1, help='Inflate the obstacles to compute unsafe states.')
        parser.add_argument('--free_padding', type=float, default=0.3, help='Ensures relatively free positions for goal and initial robot position by large inflation of the obstacles.')

        parser.add_argument('--trans_vel_low', type=float, default=0.0, help='Lower limit of desired translational velocity of robot.')
        parser.add_argument('--trans_vel_high', type=float, default=1.0, help='Upper limit of desired translational velocity of robot.')
        parser.add_argument('--std_trans_init', type=float, default=0.5, help='Initial standard deviation to use for sampling translational velocity.')
        parser.add_argument('--rot_vel_low', type=float, default=-1.0, help='Lower limit of desired rotational velocity of robot.')
        parser.add_argument('--rot_vel_high', type=float, default=1.0, help='Upper limit of desired rotational velocity of robot.')
        parser.add_argument('--std_rot_init', type=float, default=0.75, help='Initial standard deviation to use for sampling rotational velocity.')

        parser.add_argument('--action_duration', type=float, default=0.2, help='Duration(in s) for execution of each action command.')
        parser.add_argument('--max_action_count', type=int, default=300, help='Number of actions to execute before episode times out.')
        parser.add_argument('--goal_distance_tolerance', type=float, default=0.1, help='Tolerance to consider that robot has reached the goal.')
        parser.add_argument('--goal_reward', type=float, default=10.0, help='Reward received on reaching the goal.')
        parser.add_argument('--use_path_distance_reward', type=bool, default=True, help='If true, use a reward which is computed using the shortest distance from the robot\'s current position and the goal. Reward is equal to the distance the robot gets closer to the goal in a timestep.')
        parser.add_argument('--use_euclidean_distance_reward', type=bool, default=False, help='If true, use a reward which is computed using the euclidean distance from the robot\'s current position and the goal. Reward is equal to the distance the robot gets closer to the goal in a timestep.')
        parser.add_argument('--distance_reward_scaling', type=float, default=1.0, help='Scaling factor for rewards received based on distance closer to goal between consecutive time-steps.')
        parser.add_argument('--crash_reward', type=float, default=0.0, help='Additional negative reward for going into unsafe states.(Use negative reward for crashing)')
        parser.add_argument('--max_clip', type=float, default=10.0, help='Clip the maximum goal distance state and laser sensor value to this value.')
        parser.add_argument('--fov', type=float, default=180, help='Field of view to consider for the laser sensor.')
        parser.add_argument('--laser_sensor_offset', type=float, default=0.0, help='Parameter to artificially inflate the obstacles as seen by the sensor.')
        parser.add_argument('--use_min_laser_pooling', type=bool, default=True, help='If true, take minimum of the laser readings within a sector. Else, take discrete samples')

        self.args = parser.parse_args()

    def parse(self):
        return self.args
