import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int8
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from utils.common import *
from utils.robot import *


class Environment(object):

    def set_robot_pose(self, position, orientation):
        robot_pose_data = Pose2D()
        robot_pose_data.x = position[0]
        robot_pose_data.y = position[1]
        robot_pose_data.theta = orientation[2]
        self.pose_publisher.publish(robot_pose_data)

    def reset(self):
        self.is_running = True
        self.action_count = 0
        return self.get_network_state()

    def get_network_state(self):
        laser_data_states = self.get_laser_data_states()
        pose_data_states = self.get_pose_data_states()
        states = laser_data_states + pose_data_states
        if(self.use_twist_data_states):
            twist_data_states = self.get_twist_data_states()
            states += twist_data_states
        return np.asarray(states)

    def update_robot_state_data(self, msg):
        all_data = msg
        self.pose_data = all_data.pose.pose
        self.twist_data = all_data.twist.twist

    def create_position_subscriber(self):
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.update_robot_state_data)
        rospy.loginfo("Topic Subscribed: /base_pose_ground_truth")

    def update_robot_laser_data(self, msg):
        self.laser_data = msg

    def update_robot_stall_data(self, msg):
        self.stalled = msg.data

    def get_laser_data(self):
        return self.front_laser_data.ranges, self.rear_laser_data.ranges

    def get_laser_data_states(self):
        if(self.use_min_laser_pooling):
            laser_data_states = do_linear_transform(np.array( [min(self.laser_data.ranges[current: current+self.laser_slice]) for current in xrange(self.laser_slice_offset, len(self.laser_data.ranges) - self.laser_slice_offset, self.laser_slice)]) - self.laser_sensor_offset, self.max_clip, self.inverse_distance_states)
        else:
            laser_data_states = do_linear_transform(np.array(self.laser_data.ranges[self.laser_slice_offset+int(self.laser_slice/2):self.total_laser_samples - self.laser_slice_offset:self.laser_slice]) - self.laser_sensor_offset, self.max_clip, self.inverse_distance_states)
        return list(laser_data_states)

    def set_distance_map(self, distance_map):
        self.distance_map = distance_map

    def set_obstacles_map(self, obstacles_map, resolution):
        self.obstacles_map = obstacles_map
        self.resolution = resolution

    def set_goal(self, goal):
        goal_position = goal[0]
        goal_orientation = goal[1]
        goal_orientation = quaternion_from_euler(goal_orientation[0], goal_orientation[1], goal_orientation[2])
        self.goal.position = Point(goal_position[0], goal_position[1], goal_position[2])
        self.goal.orientation = Quaternion(goal_orientation[0], goal_orientation[1], goal_orientation[2], goal_orientation[3])

        self.goal_reached = False
        self.crashed = False
        self.orientation_with_goal = np.absolute(get_relative_orientation_with_goal(self.pose_data.orientation, self.goal.orientation))
        self.euclidean_distance_to_goal = get_distance(self.pose_data.position, self.goal.position)

        if(self.use_path_distance_reward):
            position_x_shifted_scaled = int(np.around((self.pose_data.position.x + self.map_size)/self.resolution))
            position_y_shifted_scaled = int(np.around((self.pose_data.position.y + self.map_size)/self.resolution))
            self.path_distance_to_goal = self.distance_map[position_x_shifted_scaled, position_y_shifted_scaled]

    def get_pose_data_states(self):
        position_data_state = do_linear_transform(get_distance(self.pose_data.position, self.goal.position), self.max_clip, self.inverse_distance_states)
        orientation_to_goal_data_state = get_relative_angle_to_goal(self.pose_data.position, self.pose_data.orientation, self.goal.position)/np.pi
        orientation_with_goal_data_state = get_relative_orientation_with_goal(self.pose_data.orientation, self.goal.orientation)/np.pi
        return [orientation_to_goal_data_state] + [position_data_state]

    def get_twist_data_states(self):
        trans_vel_state = (2*self.twist_data.linear.x - (self.v_lims[0] + self.v_lims[1]))/(self.v_lims[1] - self.v_lims[0])
        rot_vel_state = (2*self.twist_data.angular.z - (self.w_lims[0] + self.w_lims[1]))/(self.w_lims[1] - self.w_lims[0])
        return [trans_vel_state, rot_vel_state]

    def __init__(self, args, n_states):

        rospy.loginfo("Initializing Robot...")
        self.action_count = 0
        self.action_duration = args.action_duration
        self.map_size = args.map_size
        self.v_lims = [args.trans_vel_low, args.trans_vel_high]
        self.w_lims = [args.rot_vel_low, args.rot_vel_high]
        self.use_safety_cost = args.use_safety_cost
        self.crash_reward = args.crash_reward
        self.is_running = False
        self.goal = Pose()
        self.goal_distance_tolerance = args.goal_distance_tolerance
        self.goal_reward = args.goal_reward
        self.distance_reward_scaling = args.distance_reward_scaling
        self.goal_reached = False
        self.pose_data = Pose()
        self.max_clip = args.max_clip
        self.inverse_distance_states = True
        self.twist_data = Twist()
        self.network_laser_inputs = n_states - 2
        self.use_twist_data_states = False
        self.use_path_distance_reward = args.use_path_distance_reward
        self.use_euclidean_distance_reward = args.use_euclidean_distance_reward
        self.fov = args.fov
        if(self.use_twist_data_states):
            self.network_laser_inputs -= 2
        self.use_min_laser_pooling = args.use_min_laser_pooling
        self.laser_sensor_offset = args.laser_sensor_offset
        self.motion_command = Twist()
        self.stalled = False

        self.max_action_count = args.max_action_count


        rospy.init_node('ros_node')
        rospy.loginfo("Node Created")

        rospy.loginfo("Examining Laser Sensor...")
        ##Added for single laser(stage)
        self.laser_data = LaserScan()
        msg = rospy.wait_for_message("/base_scan", LaserScan, 10)
        self.total_laser_samples = len(msg.ranges)
        self.laser_slice = int((self.fov*self.total_laser_samples)/(270*self.network_laser_inputs))
        self.laser_slice_offset = int((self.total_laser_samples*(270-self.fov))/(2*270))

        rospy.loginfo("Setting Publishers...")

        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size = 1) ##Set queue size
        rospy.loginfo("Publisher Created: /cmd_vel")

        self.pose_publisher = rospy.Publisher('/cmd_pose', Pose2D, queue_size = 1)
        rospy.loginfo("Publisher Created: /cmd_pose")

        ##Added for single laser(stage)
        rospy.loginfo("Setting Subscribers...")
        rospy.Subscriber("/base_scan", LaserScan, self.update_robot_laser_data)

        rospy.Subscriber("/stalled", Int8, self.update_robot_stall_data)

        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.update_robot_state_data)
        rospy.loginfo("Topic Subscribed: /base_pose_ground_truth")

        rospy.sleep(5)

    def execute_action(self, action):

        self.action_count += 1
        self.motion_command = Twist()

        self.motion_command.linear.x = action[0]
        self.motion_command.angular.z = action[1]

        action_start_time = rospy.Time.now()
        flag = False

        reward = 0
        safety_cost = 0
        crashed = False

        while(rospy.Time.now() - action_start_time < rospy.Duration(self.action_duration)):
            if(rospy.Time.now() - action_start_time < rospy.Duration(0)): #Stage simulator crashes and restarts sometimes, the flag gets activated in that case to re-run the episode.
                #print("Simulator crashed!!!!")
                flag = True
                self.is_running = False
                break
            self.velocity_publisher.publish(self.motion_command)
            if(self.stalled):
                crashed = True

        if(crashed):
            self.crashed = True
            safety_cost += 1
            reward += self.crash_reward

        next_state = self.get_network_state()

        if(self.action_count >= self.max_action_count):
            self.is_running = False

        next_euclidean_distance_to_goal = get_distance(self.pose_data.position, self.goal.position)

        if(self.use_euclidean_distance_reward):
            reward += -self.distance_reward_scaling*(next_euclidean_distance_to_goal - self.euclidean_distance_to_goal)

        if(self.use_path_distance_reward):
            position_x_shifted_scaled = int(np.around((self.pose_data.position.x + self.map_size)/self.resolution))
            position_y_shifted_scaled = int(np.around((self.pose_data.position.y + self.map_size)/self.resolution))
            next_path_distance_to_goal = self.distance_map[position_x_shifted_scaled, position_y_shifted_scaled]
            if(next_path_distance_to_goal != np.inf and self.path_distance_to_goal != np.inf):
                reward += -self.distance_reward_scaling*(next_path_distance_to_goal - self.path_distance_to_goal)
            self.path_distance_to_goal = next_path_distance_to_goal

        if(next_euclidean_distance_to_goal < self.goal_distance_tolerance):
            self.goal_reached = True
            self.is_running = False
            reward += self.goal_reward

        self.euclidean_distance_to_goal = next_euclidean_distance_to_goal

        if(self.use_safety_cost):
            return next_state, reward, safety_cost, flag, not(self.is_running)

        else:
            return next_state, reward, flag, not(self.is_running)

    def get_pose(self):
        return self.pose_data

    def get_position(self):
        return self.pose_data.position

    def get_orientation(self):
        return self.pose_data.orientation

    def get_goal_pose(self):
        return self.goal

    def get_goal_position(self):
        return self.goal.position

    def get_linear_speed(self):
        linear_velocity = self.twist_data.linear
        return np.sqrt(linear_velocity.x**2 + linear_velocity.y**2)

    def get_angular_speed(self):
        return np.absolute(self.twist_data.angular.z)
