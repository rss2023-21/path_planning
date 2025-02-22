#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory, PriorityQueue, SquareGrid
import matplotlib.pyplot as plt
# from skimage.morphology import disk, dilation
import tf

# polyfill for skimage.morphology
def create_disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = x**2 + y**2 <= radius**2
    return disk.astype(np.uint8)

def dilation(image, se):
    padded_image = np.pad(image, ((se.shape[0]//2, se.shape[0]//2), (se.shape[1]//2, se.shape[1]//2)), mode='constant')
    result = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            roi = padded_image[y:y + se.shape[0], x:x + se.shape[1]]
            result[y, x] = np.max(roi * se)

    return result


class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    GRID_MAP_SCALE = 5  # number of cells in grid map vs. actual map

    def __init__(self):
        self.odom_topic = "/pf/pose/odom" # rospy.get_param("~odom_topic")
        rospy.loginfo("Using odometry topic: %s", self.odom_topic)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

        self.map = None



    def map_cb(self, msg):
        rospy.loginfo("Received map!")
        rospy.loginfo("Map info: %d x %d, %f m/cell", msg.info.width, msg.info.height, msg.info.resolution)

        # transform that converts from map coordinates to grid coordinates
        # Suppose the message received on this channel is called msg: to convert from pixel coordinates (u, v) to real coordinates (x, y), you should multiply (u, v) through by msg.info.resolution, and then apply the rotation and translation specified by msg.info.origin.orientation and msg.info.origin.position. To convert the other way (x, y) -> (u, v), reverse these operations. 

        self.rotation = np.array([msg.info.origin.orientation.x, msg.info.origin.orientation.y, msg.info.origin.orientation.z, msg.info.origin.orientation.w])
        self.rotation = tf.transformations.euler_from_quaternion(self.rotation)
        theta = self.rotation[2]
        self.rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        self.translation = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.resolution = msg.info.resolution

        rospy.loginfo("Rotation: %s", self.rotation)
        rospy.loginfo("Translation: %s", self.translation)
        rospy.loginfo("Resolution: %f", self.resolution)
        rospy.loginfo("Rotation Matrix:")
        rospy.loginfo(self.rot_mat)


        # convert map to numpy array
        m = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        

        # imshow and save image to map.png
        # fig, ax = plt.subplots()
        # ax.imshow(m)
        # fig.savefig("/home/racecar/racecar_ws/src/path_planning/map.png")

        # morphological dilations using skimage.morphology.disk
        # this makes the walls thicker to stop path planning from getting too close to the walls
        # m2 = dilation(m, disk(5))

        m2 = dilation(m, create_disk(5))

        # fig, ax = plt.subplots()
        # ax.imshow(m2)
        # fig.savefig("/home/racecar/racecar_ws/src/path_planning/map2.png")

        self.map = m2

        # downsample the map to create a search grid
        self.grid = SquareGrid(m2.shape[0] // self.GRID_MAP_SCALE, m2.shape[1] // self.GRID_MAP_SCALE)
        grid_downsampled = m2[::self.GRID_MAP_SCALE, ::self.GRID_MAP_SCALE]

        # rospy.loginfo(np.unique(grid_downsampled))

        # fig, ax = plt.subplots()
        # ax.imshow(grid_downsampled)
        # fig.savefig("/home/racecar/racecar_ws/src/path_planning/grid_downsampled.png")

        self.grid.walls = set([(u, v) for u in range(grid_downsampled.shape[0]) for v in range(grid_downsampled.shape[1]) if grid_downsampled[u, v] == 100])

    def odom_cb(self, msg):
        # rospy.loginfo("Received odometry!")
        # update start pose from odometry
        self.start_pose = msg.pose.pose

        # get x and y from pose
        self.start = Point(self.start_pose.position.x, self.start_pose.position.y, 0)
        

    def goal_cb(self, msg):
        while self.map is None:
            # rospy.loginfo("No map received yet!")
            # return
            pass
        
        rospy.loginfo("Received goal!")
        # update goal pose
        self.goal_pose = msg.pose

        # get x and y from pose
        self.goal = Point(self.goal_pose.position.x, self.goal_pose.position.y, 0)

        # plan path
        self.plan_path(self.start, self.goal, self.map)

    def point_to_grid_loc(self, point):
        """ Convert point in real world coordinates to grid location """
        #Add rotation

        point_numpy = np.array([point.x, point.y]) - self.translation
        # rospy.loginfo("pre-rotation")
        # rospy.loginfo(point_numpy)

        #inverse rotate
        point_numpy = np.matmul(np.linalg.inv(self.rot_mat), point_numpy) #p = R^-1*p'
        # rospy.loginfo("post-rotation")
        # rospy.loginfo(point_numpy)
        
        point_numpy /= self.resolution
        point_numpy /= self.GRID_MAP_SCALE
        # remember to swap coords!
        grid_loc = (int(point_numpy[1]), int(point_numpy[0]))
        # rospy.loginfo(grid_loc)
        return grid_loc

    def grid_loc_to_point(self, grid_loc):
        #Add rotation
        """ Convert grid location to point in real world coordinates """
        point_numpy = np.array(grid_loc)[::-1] * self.GRID_MAP_SCALE
        point_numpy = point_numpy.astype(float)
        point_numpy *= self.resolution
        point_numpy = np.matmul(self.rot_mat, point_numpy) #p' = R*p
        point_numpy += self.translation

        return Point(point_numpy[0], point_numpy[1], 0)

    def plan_path(self, start_point, end_point, map):
        rospy.loginfo('Planning path from (%f, %f) to (%f, %f)', start_point.x, start_point.y, end_point.x, end_point.y)
        start = self.point_to_grid_loc(start_point)
        goal = self.point_to_grid_loc(end_point)
        rospy.loginfo("start")
        rospy.loginfo(start)
        rospy.loginfo("goal")
        rospy.loginfo(goal)

        # time how long it takes to plan path
        start_time = time.time()

        # Sample code from https://www.redblobgames.com/pathfinding/a-star/
        def heuristic(a, b):
            # return 0
            (x1, y1) = a
            (x2, y2) = b
            if a in self.grid.walls or b in self.grid.walls:
                return 1000000
            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # return abs(x1 - x2) + abs(y1 - y2)
            
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        
        while not frontier.empty():
            current = frontier.get()
            
            if current == goal:
                break
            
            for next in self.grid.neighbors(current):
                new_cost = cost_so_far[current] + self.grid.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(next, goal)
                    frontier.put(next, priority)
                    came_from[next] = current
    
        # self.trajectory.addPoint(start_point)
        # rospy.loginfo(came_from)
        points_reversed = []
        while current != start:
            # rospy.loginfo((current, cost_so_far[current]))
            points_reversed.append(self.grid_loc_to_point(current))
            current = came_from[current]

        for point in points_reversed[::-1]:
            self.trajectory.addPoint(point)

        # self.trajectory.addPoint(end_point)
	    # rospy.loginfo("The trajectory")
	    # rospy.loginfo(self.trajectory.points)
        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()

        # # create an array with size of the grid, and put the cost of each cell in the array using cost_so_far
        # # fill array with 
        # cost_array = np.zeros((self.grid.width, self.grid.height)) + 200
        # for cell in cost_so_far:
        #     cost_array[cell[0]][cell[1]] = cost_so_far[cell]

        # # save colorbar to cost_array_colorbar.png
        # fig, ax = plt.subplots()
        # im = ax.imshow(cost_array)
        # fig.colorbar(im)
        # fig.savefig("/home/racecar/racecar_ws/src/path_planning/cost_array_colorbar.png")

        # fig, ax = plt.subplots()
        # im = ax.imshow(path_array)
        # # fig.colorbar(im)
        # fig.savefig("/home/racecar/racecar_ws/src/path_planning/path_array.png")

        # # 

        # rospy.loginfo(start)
        # rospy.loginfo(goal)
        # rospy.loginfo(self.grid.walls)


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
