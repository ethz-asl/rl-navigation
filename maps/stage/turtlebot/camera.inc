
define turtlebot_camera camera
(
  resolution [640 480]
  fov [70 50]

  size [0.1 0.07 0.05]

	block( points 4
				 point[0] [0 0]
				 point[1] [0 1]
				 point[2] [1 1]
				 point[3] [1 0]
				 z [0 0.21]
				 )
)
