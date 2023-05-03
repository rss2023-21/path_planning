#!/usr/bin/env python
from __future__ import division
import rospy
import numpy as np
import time
import utils
import tf
import math
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Float32

from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        # self.odom_topic       = rospy.get_param("~odom_topic")
        self.lookahead        = 0.6# FILL IN 
        self.default_speed = 0.1
        self.speed            = 0.1 # FILL IN 
        self.wheelbase_length = .35 # FILL IN
        self.steering_constant = .35
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)
        self.odom_sub  = rospy.Subscriber('/odom', Odometry,
                                          self.odom_callback,
                                          queue_size=1)
        
        self.pos1_publisher = rospy.Publisher('/pose1', PoseStamped, queue_size=1)
        self.pos2_publisher = rospy.Publisher('/pose2', PoseStamped, queue_size=1)
        self.error_pub = rospy.Publisher('/err', Float32, queue_size=1)
        self.error_pub2 = rospy.Publisher('/err2', Float32, queue_size=1)

    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        rospy.logerr("Receiving new trajectory: %d points", len(msg.poses))
	#rospy.logerr(len(msg.poses))
	#rospy.logerr("points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

    def minimum_distance(self,v, w, p):
        # Calculate the squared length of the line segment vw
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / ((w[0] - v[0]) ** 2 + (w[1] - v[1]) ** 2)))
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        return math.sqrt((p[0] - projection[0]) ** 2 + (p[1] - projection[1]) ** 2) 
    
    
    def find_circle_line_intersection(self, center, radius, startPoint, endPoint):
        Q = center
        r = radius
        P1 = startPoint      # Start of line segment
        V = endPoint - startPoint # Vector along line segment
        
        
        a = V.dot(V)
        b = 2 * V.dot(P1 - Q)
        c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r**2

        disc = b**2 - 4 * a * c
        if disc < 0:
            return None
        
        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        
        #rospy.logerr(str(t1) + '||' + str(t2))
        
        # if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
        #     return None
        
        return ( P1 + t1 * V if (0 <= t1 <= 1) else None, 
                 P1 + t2 * V if (0 <= t1 <= 1) else None )
    
    
    def checkInFront(self, point, robotLoc, theta):
        if point is None:
            return point
        directionVec = np.array([math.cos(theta), math.sin(theta)])
        differenceVec = point - robotLoc
        dot_prod = np.dot(directionVec, differenceVec)
        if (dot_prod >= -1):
            return point
        return None
    
    
    def odom_callback(self, msg):
        xLoc = msg.pose.pose.position.x
        yLoc = msg.pose.pose.position.y
        curLoc = np.array([xLoc, yLoc])
        #rospy.loginfo('hi')
        nearestPoints = []
        for i in range(len(self.trajectory.points) - 1):
             nearestPoints.append(self.minimum_distance(self.trajectory.points[i],
                                                             self.trajectory.points[i+1], 
                                                             (xLoc, yLoc)))

        #rospy.logerr("odom comin")
        # if (len(nearestPoints) > 0):
        # rospy.loginfo(self.trajectory.points)
        
        if (len(nearestPoints) <= 0):
            rospy.logerr('no trajectory loaded yet')
            self.speed = 0
            drive_cmd = AckermannDriveStamped()
            drive_cmd.header.stamp = rospy.Time.now()
            drive_cmd.drive.steering_angle = 0
            drive_cmd.drive.speed = self.speed

            self.drive_pub.publish(drive_cmd)
            return
          
        nearSegmentIndex = np.argmin((nearestPoints))
        rospy.logerr("Num of nearest points: %d", len(nearestPoints))
        rospy.logerr("Num of trajectory points: %d", len(self.trajectory.points))
        
        onePoint = None
        otherPoint = None
        i = nearSegmentIndex
        rospy.logerr("Current index: %d", i)
        rospy.logerr("Current distance value: %d", len(self.trajectory.points) - i ) 

        while(onePoint is None and otherPoint is None):
            if (i >= len(self.trajectory.points) - 1):
                rospy.logerr(str(nearSegmentIndex) + ' no find path')
                self.speed = 0
                break
            
            #rospy.logerr("Index in while loop: %d", i)
            if (len(self.trajectory.points) - nearSegmentIndex < 30):
                self.speed = 0
                drive_cmd = AckermannDriveStamped()
                drive_cmd.header.stamp = rospy.Time.now()
                drive_cmd.drive.steering_angle = 0
                drive_cmd.drive.speed = self.speed
                rospy.logerr('STOPPING, WITHIN LOOKAHEAD DISTANCE')
                self.drive_pub.publish(drive_cmd)
                break
            
            #rospy.logwarn("just seeing if it continued after the break")
            startPoint = self.trajectory.points[i]
            endPoint = self.trajectory.points[i+1]
            points = self.find_circle_line_intersection(curLoc,self.lookahead,np.array(startPoint),np.array(endPoint))

            if points is None:
                self.speed = 0
                drive_cmd = AckermannDriveStamped()
                drive_cmd.header.stamp = rospy.Time.now()
                drive_cmd.drive.steering_angle = 0
                drive_cmd.drive.speed = self.speed
                rospy.logerr('points are none')
                self.drive_pub.publish(drive_cmd)
                return

            onePoint, otherPoint = points[0], points[1]
            no, yes, theta = euler_from_quaternion((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
            onePoint = self.checkInFront(onePoint,curLoc, theta)
            otherPoint = self.checkInFront(otherPoint,curLoc, theta)
            i += 1
            self.speed = self.default_speed
        
        
        if not onePoint is None:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = '/map'
            pose.pose.position.x = onePoint[0]
            pose.pose.position.y = onePoint[1]
            self.pos1_publisher.publish(pose)
        
        if not otherPoint is None:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = '/map'
            pose.pose.position.x = otherPoint[0]
            pose.pose.position.y = otherPoint[1]
            self.pos2_publisher.publish(pose)
        
        if not onePoint is None:
            rospy.logerr("smths moving")
            self.driveCommand(curLoc - onePoint, theta)
            return
        
        
        if not otherPoint is None:
            rospy.logerr("Other thing is moving")
            self.driveCommand(curLoc - otherPoint, theta)
            return
    
    def driveCommand(self, point, theta):
        #rospy.logerr("Sending drive command")
	    #rospy.logerr(str(point[0]) + ' ' +  str(point[1]))
        drive_cmd = AckermannDriveStamped()
    
        
        nu = math.atan2(point[1], point[0])
        nu = (nu - theta) % math.pi
        if nu > 2:
            self.error_pub.publish(math.pi - nu)
        else:
           self.error_pub.publish(nu) 
        drive_angle = (2*self.wheelbase_length*math.cos(nu)/self.lookahead)
        drive_angle *= self.steering_constant
        self.error_pub2.publish(abs(drive_angle))
        # rospy.logerr(str(drive_angle))
        drive_cmd.header.stamp = rospy.Time.now()
        drive_cmd.drive.steering_angle = drive_angle
        drive_cmd.drive.speed = self.speed
        

        self.drive_pub.publish(drive_cmd)
    
         
    def my_distance(self, x,y):
        return math.sqrt(x**2 + y**2 )
        
        

if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    # rospy.loginfo('hi2')
    pf = PurePursuit()
    rospy.spin()
