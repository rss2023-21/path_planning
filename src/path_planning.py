#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
import matplotlib.pyplot as plt
from skimage.morphology import disk, dilation

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)


    def map_cb(self, msg):
        rospy.loginfo("Received map!")
        rospy.loginfo("Map info: %d x %d, %f m/cell", msg.info.width, msg.info.height, msg.info.resolution)

        # convert map to numpy array
        m = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # imshow and save image to map.png
        fig, ax = plt.subplots()
        ax.imshow(m)
        fig.savefig("/home/racecar/racecar_ws/src/path_planning/map.png")

        # morphological dilations using skimage.morphology.disk
        # this makes the walls thicker to stop path planning from getting too close to the walls
        m2 = dilation(m, disk(5))
        fig, ax = plt.subplots()
        ax.imshow(m2)
        fig.savefig("/home/racecar/racecar_ws/src/path_planning/map2.png")

        self.map = m2

    def odom_cb(self, msg):
        rospy.loginfo("Received odometry!")
        # update start pose from odometry
        self.start_pose = msg.pose.pose

        # get x and y from pose
        self.start = Point(self.start_pose.position.x, self.start_pose.position.y)
        

    def goal_cb(self, msg):
        rospy.loginfo("Received goal!")
        # update goal pose
        self.goal_pose = msg.pose

        # get x and y from pose
        self.goal = Point(self.goal_pose.position.x, self.goal_pose.position.y)

        # plan path
        self.plan_path(self.start, self.goal, self.map)

    def plan_path(self, start_point, end_point, map):
        rospy.loginfo('Planning path from (%f, %f) to (%f, %f)', start_point.x, start_point.y, end_point.x, end_point.y)

        self.trajectory.addPoint(start_point)
        self.trajectory.addPoint(end_point)

        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
