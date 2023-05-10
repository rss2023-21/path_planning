#!/usr/bin/env python2

import sys
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

if __name__ == "__main__":
    rospy.init_node("goal_pub")

    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1, latch=True)
    msg = PoseStamped()
    msg.pose.position.x = 2.53 #1.35775518417 #-24.004
    msg.pose.position.y = 5.29 #5.38434696198 #-.0857
    msg.pose.orientation.x = 0
    msg.pose.orientation.y = 0
    msg.pose.orientation.z = 0.738
    msg.pose.orientation.w =  0.6729

    pub.publish(msg)
    rospy.spin()
