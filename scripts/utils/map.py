import numpy as np
import heapq
import rospkg
import os

def get_obstacles_map(map_limit, obstacles, resolution, padding = 0.3):
#Computes grid-occupancy of the whole map using the obstacle positions and uses padding to inflate the obstacles to use
    map_size = 2*int(map_limit/resolution) + 1
    obstacles_map = np.zeros((map_size, map_size))
    for x,y in obstacles:
        x_shifted_scaled_low = int(np.floor((x + map_limit)/resolution))
        x_shifted_scaled_high = int(np.ceil((x + map_limit)/resolution))
        y_shifted_scaled_low = int(np.floor((y + map_limit)/resolution))
        y_shifted_scaled_high = int(np.ceil((y + map_limit)/resolution))
        obstacles_map[x_shifted_scaled_low:x_shifted_scaled_high+1, y_shifted_scaled_low:y_shifted_scaled_high+1] = 1
    obstacles_map_copy = np.copy(obstacles_map)
    if(resolution <= padding):
        extra_positions = int(np.ceil(padding/resolution))
        for i in range(map_size):
            for j in range(map_size):
                if obstacles_map[i,j] == 1:
                    obstacles_map_copy[np.max((0, i - extra_positions)): np.min((i + extra_positions + 1, map_size)), np.max((0, j- extra_positions)): np.min((j + extra_positions +1, map_size))] = 1
    return obstacles_map_copy


def get_distance_map(map_limit, obstacles_map, goal, resolution):
#Computes the shortest path distance from the goal to each point in the map using Djikstra

    map_size = 2*int(map_limit/resolution) + 1

    value_map = np.ones((map_size, map_size)) * np.inf
    visited_cell = np.zeros((map_size, map_size), dtype=np.bool)
    opened_cell = np.zeros((map_size, map_size), dtype=np.bool)

    goal_shifted_scaled = (int(np.around((goal[0] + map_limit)/resolution)), int(np.around((goal[1] + map_limit)/resolution)))

    value_map[goal_shifted_scaled[0], goal_shifted_scaled[1]] = 0

    cell_queue = []
    cell_pointer = {}
    entry = (0, goal_shifted_scaled)
    cell_pointer[goal_shifted_scaled] = entry
    heapq.heappush(cell_queue, entry)

    while cell_queue:
        currentCell = heapq.heappop(cell_queue)
        x = currentCell[1][0]
        y = currentCell[1][1]
        visited_cell[x, y] = True
        current_value = value_map[x, y]

        for xi in xrange(max(0, x - 1), min(map_size, x + 2)):
            for yi in xrange(max(0, y - 1), min(map_size, y + 2)):
                if not visited_cell[xi, yi] and not obstacles_map[xi, yi]:
                    cost = ((x - xi) ** 2 + (y - yi) ** 2) ** 0.5
                    if (current_value + cost < value_map[xi, yi]):
                        value_map[xi, yi] = current_value + cost
                        entry = (current_value + cost, (xi, yi))
                        if not opened_cell[xi, yi]:
                                heapq.heappush(cell_queue, entry)
                                opened_cell[xi, yi] = True
                        else:
                            cell_queue.remove(cell_pointer[(xi, yi)])
                            heapq.heappush(cell_queue, entry)
                        cell_pointer[(xi, yi)] = entry
    return resolution*value_map

def get_obstacle_positions(map_limit, map_name):
#Load the file which contains all the obstacle positions
    return list(np.load(os.path.join(rospkg.RosPack().get_path('reinforcement_learning_navigation'),'maps', map_name+'.npy')))

def get_map_choice(map_strategy):
#Choose between randomly selecting a map or using a specific map
    if(map_strategy == 'random_sampling'):
        return np.random.choice(np.arange(3))
    elif(map_strategy == 'map-1'):
        return 0
    elif(map_strategy == 'map-2'):
        return 1
    elif(map_strategy == 'map-3'):
        return 2
    else:
        return 3

def get_free_position(free_map, map_resolution, map_limit, map_case):
#Sample a point in the map which is fairly away from obstacles to set the goal or initial robot position
    while(True):
        free_position_0 = np.random.randint(0, 2*int(map_limit/map_resolution) + 1)
        free_position_1 = np.random.randint(0, 2*int(map_limit/map_resolution) + 1)
        if(map_case == 0):
            free_position_1 += 100
        elif(map_case == 1):
            free_position_0 += 100
            free_position_1 += 100
        elif(map_case == 2):
            free_position_0 += 100
        else:
            pass
        if(free_map[free_position_0][free_position_1] == 0):
            break
    return [np.float(free_position_0)*map_resolution - 2*map_limit, np.float(free_position_1)*map_resolution - 2*map_limit]
