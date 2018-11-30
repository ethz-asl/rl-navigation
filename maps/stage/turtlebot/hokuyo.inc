define hokuyo_common ranger
(
	sensor( 			
    # laser-specific properties
    range [ 0.0  60.0 ]
    fov 270.0
    samples 1080
  )
  # generic model properties
  color "blue"
  size [ 0.07 0.07 0.05 ] # dimensions from LMS200 data sheet	
)

define hokuyo_base hokuyo_common
(
	block( points 4
				 point[0] [0 0]
				 point[1] [0 1]
				 point[2] [1 1]
				 point[3] [1 0]
				 z [0 0.21]
				 )
)

define hokuyolaser hokuyo_base()
