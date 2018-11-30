include "turtlebot/hokuyo.inc"
include "turtlebot/camera.inc"

define turtlebot_base position 
(
  color "red"			            # Default color.
  gui_nose 1                  	# Draw a nose on the robot so we can see which way it points
  obstacle_return 1           	# Can hit things.
  ranger_return 0.5            	# reflects sonar beams
  blob_return 1               	# Seen by blobfinders  
  fiducial_return 1           	# Seen as "1" fiducial finders

  localization "gps"         	  
  localization_origin [0 0 0 0] 	# Start odometry at (0, 0, 0).

  # alternative odometric localization with simple error model
  # localization "odom"         		# Change to "gps" to have impossibly perfect, global odometry
  #  odom_error [0.03 0.03 999999 999999 999999 0.02]   	# Odometry error or slip in X, Y and Theta
                                   	# (Uniform random distribution)   

  # four DOF kinematics limits
  # [ xmin xmax ymin ymax zmin zmax amin amax ]				
  velocity_bounds [-0.7 0.7 0 0 0 0 -180.0 180.0 ]					
  acceleration_bounds [-2.0 2.0 0 0 0 0 -120.0 120.0 ]

  # actual size
  size [0.356 0.356 0.109]

  # the center of rotation is offset from its center of area
  origin [0 0 0 0]

  # draw a nose on the robot so we can see which way it points
  gui_nose 1

  # estimated mass in KG
  mass 2.4
  # differential steering model
  drive "diff"
)

define turtlebot_base_sensors turtlebot_base
(
  hokuyolaser( pose [0.13 0 0 0])
  #turtlebot_camera( pose [0.13 0 0.05 0])
)

define turtlebot turtlebot_base_sensors
(  
  size [0.356 0.356 0.109]
  size [0.356 0.356 0.109]
  # this block approximates the circular shape of a Roomba
  block(
    points 16
    point[0] [ 0.225 0.000 ]
    point[1] [ 0.208 0.086 ]
    point[2] [ 0.159 0.159 ]
    point[3] [ 0.086 0.208 ]
    point[4] [ 0.000 0.225 ]
    point[5] [ -0.086 0.208 ]
    point[6] [ -0.159 0.159 ]
    point[7] [ -0.208 0.086 ]
    point[8] [ -0.225 0.000 ]
    point[9] [ -0.208 -0.086 ]
    point[10] [ -0.159 -0.159 ]
    point[11] [ -0.086 -0.208 ]
    point[12] [ -0.000 -0.225 ]
    point[13] [ 0.086 -0.208 ]
    point[14] [ 0.159 -0.159 ]
    point[15] [ 0.208 -0.086 ]
    z [0 0.22]
  )
)
