#!/usr/bin/env python2

import sys
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

if __name__ == "__main__":
    rospy.init_node("goal_pub")

    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1, latch=True)
    msg = PoseStamped()
    msg.pose.position.x = -24.004
    msg.pose.position.y = -.0857
    msg.pose.orientation.x = 0
    msg.pose.orientation.y = 0
    msg.pose.orientation.z = 0.999670
    msg.pose.orientation.w = .02565789

    pub.publish(msg)
    rospy.spin()