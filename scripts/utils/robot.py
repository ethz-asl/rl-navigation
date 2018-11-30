import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def do_linear_transform(distance, max_clip = 20.0, inverse = True):
    normalized_distance = np.minimum(distance, max_clip)/max_clip
    if(inverse):
        return 1 - 2*normalized_distance
    else:
        return normalized_distance

def get_distance(pos1, pos2):
    return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)

def get_angle(pos1, pos2):
    return np.arctan2(pos2.y - pos1.y, pos2.x - pos1.x)

def get_relative_orientation_with_goal(orientation, goal_orientation):
    angle = euler_from_quaternion([goal_orientation.x, goal_orientation.y, goal_orientation.z, goal_orientation.w])[2] - euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])[2]
    if(angle/np.pi < -1):
        return angle + 2*np.pi
    elif(angle/np.pi > 1):
        return angle - 2*np.pi
    return angle

def get_relative_angle_to_goal(position, orientation, goal_position):
    angle = get_angle(position, goal_position) - euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])[2]
    if(angle/np.pi < -1):
        return angle + 2*np.pi
    elif(angle/np.pi > 1):
        return angle - 2*np.pi
    return angle
