std_msgs/Header header
geometry_msgs/Point[] points
# orientations[i] is the desired shear orientation at points[i].
# If all components are zero the control stack uses identity (unoriented cut).
# Populated when WP2 branch-vector-field orientation is available.
geometry_msgs/Quaternion[] orientations
