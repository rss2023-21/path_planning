#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel


from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion

from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
import tf
import math

# import scipy

import numpy as np


class ParticleFilter:
    initialized = False;
    lock = False
    particles = None
    debug = False
    
    def __init__(self):
        # Get parameters
        # if self.debug: rospy.loginfo('starting init')
        self.lock = True
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")

        self.N = rospy.get_param("~num_particles")
        self.downSampleSize = 100

        # initialize particle positions using random gaussian distribution x: 0.01, y: 0.01, theta: 0.001
        self.particles = np.random.multivariate_normal([0, 0, 0], [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.001]], self.N)
        self.motion_model = MotionModel()
        self.last_time = 0
				
        self.num_samples = 10 #TODO: Tune this number
        
        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.lidar_callback, # TODO: Fill this in
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odom_callback,
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.initialize_pose_callback, # TODO: Fill this in
                                          queue_size=1)
        
        self.odom_sub = rospy.Subscriber('/odom', Pose,
                                         self.print_data, queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        self.particle_pub  = rospy.Publisher("/particles", PoseArray, queue_size = 1)
        self.dif_pub  = rospy.Publisher("/dif", Float32, queue_size = 1)
        
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()
        
        
        self.latest_scan = 0
        # rospy.loginfo(self.sensor_model.evaluate)
        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.angles = None
        self.compressed_angles = None
        self.initialized = True
        self.average_poses = []
        self.lock = False
        # if self.debug: rospy.loginfo('ending init')

    def print_data(self,msg):
        x_pos = msg.pose.pose.position.x
        y_pos = msg.pose.pose.position.y
        
        x_mean = np.mean(self.particles[:, 0])
        y_mean = np.mean(self.particles[:, 1])
        
        x_dif = x_mean - x_pos
        y_dif = y_mean - y_pos
        abs_dif = math.sqrt((x_dif**2) + (y_dif**2))
        
        self.dif_pub.publish(abs_dif)
        
        
        
        
    def publish_odometry(self):
        x_mean = np.mean(self.particles[:, 0])
        y_mean = np.mean(self.particles[:, 1])
        theta_mean = np.angle(np.sum(np.exp(1j * self.particles[:, 2])))
        # rospy.loginfo(str(x_mean) + ' ' + str(y_mean) + ' ' + str(theta_mean))
        
        ret = Odometry()
        ret.header.stamp = rospy.Time.now()
        ret.header.frame_id = "/base_link_pf"
    
        ret.pose.pose.position.x = x_mean
        ret.pose.pose.position.y = y_mean
        ret.pose.pose.position.z = 0.0

        quat = tf.tranformations.quartenion_from_euler(0, 0, theta_mean)
        ret.pose.pose.orientation.x = quat[0]
        ret.pose.pose.orientation.y = quat[1]
        ret.pose.pose.orientation.z = quat[2]
        ret.pose.pose.orientation.w = quat[3]
        self.odom_pub.publish(ret)
        


        # some odometry message
        br = tf.TransformBroadcaster()
        br.sendTransform((x_mean, y_mean, 0),
                         tf.transformations.quaternion_from_euler(0, 0, theta_mean),
                         rospy.Time.now(),
                         '/base_link_pf',
                         "/map")
        
        self.publish_particle_poses()
        
    def publish_particle_poses(self):
        poseArr = []
        for i in range(len(self.particles)):
            particle = self.particles[i]
            pose = Pose()
            
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0
            
            quat = quaternion_from_euler(0,0,particle[2])
            pose.orientation = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3])
            
            poseArr.append(pose)
        
        poseArray = PoseArray()
        poseArray.header.stamp = rospy.get_rostime()
        poseArray.header.frame_id = '/map'
        poseArray.poses = poseArr
        
        self.particle_pub.publish(poseArray)
        
		
    def lidar_callback(self, msg):
        """
        recompute probabilities based on the sensor model
        """
        # if self.lock == True:
        #     return
        # self.lock = True
        #PUT THIS ALL IN THE ODOMETRY CALLBACK
        # if not self.initialized:
        #     self.lock = False
        #     return
        # if self.debug: rospy.loginfo('starting lidar callback')
        if not self.lock:
            self.latest_scan = msg.ranges
        # if self.debug: rospy.loginfo('ending lidar callback')
        # self.lock = False

    def odom_callback(self, msg):
        """
        update the particle positions based on the motion model when recieving odometry call
        """
        # if self.debug: rospy.loginfo('starting odom callback')
        if not self.initialized:
            self.lock = False
            return
        
        if self.lock == True:
            return
        
        self.lock = True
        
        #update time set
        # if self.debug: rospy.loginfo('odom callback 1')
        cur_time = msg.header.stamp.to_sec()        
        if self.last_time == 0:
            self.last_time = cur_time
        
        #get delta time
        time_passed = cur_time - self.last_time
        self.last_time = cur_time
        # if self.debug: rospy.loginfo('odom callback 2')
        # update particles from odometry model
        self.particles = self.motion_model.evaluate(self.particles, [msg.twist.twist.linear.x * time_passed , msg.twist.twist.linear.y * time_passed, msg.twist.twist.angular.z * time_passed])
        
        #get probabilities from sensor model using the latest scan
        if self.latest_scan == 0:
            self.lock = False
            return
        # rospy.loginfo('1.5')
        # if self.debug: rospy.loginfo('odom callback 3')
        probabilities = self.sensor_model.evaluate(self.particles, np.array(self.latest_scan))
        # rospy.loginfo('1.51')
        if probabilities == None:
            self.lock = False
            return
        probabilities /= probabilities.sum()  
        # resample according to the probabilities
        # if self.debug: rospy.loginfo('odom callback 4')
        particles_ind = np.random.choice([i for i in range(self.N)], size=(self.N), p=probabilities) #Length, size number of particles, p is the probabilities
        self.particles = self.particles[particles_ind]
        #publish odometry
        self.publish_odometry()
        self.lock = False
        # if self.debug: rospy.loginfo('ending odom callback')
    
    def initialize_pose(self, x, y, theta):
        """
        initialize the particles using the pose estimate
        """
        
        self.lock = True
        self.particles = np.random.multivariate_normal([x, y, theta], [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.001]], self.N)
        self.lock = False
        
    def initialize_pose_callback(self, msg):
        """
        callback for the pose initialization
        """
        # if self.debug: rospy.loginfo('starting init pose callback')
        self.lock = True
        # get angle from msg.pose.pose.orientation quaternion
        quaternion_orientation = msg.pose.pose.orientation

        # convert quaternion to euler angles
        euler_orientation = euler_from_quaternion([quaternion_orientation.x, quaternion_orientation.y, quaternion_orientation.z, quaternion_orientation.w])
        theta = euler_orientation[2]

        self.initialize_pose(msg.pose.pose.position.x, msg.pose.pose.position.y, theta)
        self.publish_odometry()
        self.lock = False
        # if self.debug: rospy.loginfo('ending init pose callback')
    # def downsample(self, laserScan):
        
    #     # return laserScan.ranges
        
    #     if self.angles == None:
    #         self.angles = np.linspace(laserScan.angle_min, laserScan.angle_max, len(laserScan.ranges))
    #     if self.compressed_angles == None:
    #         self.compressed_angles = np.linspace(laserScan.angle_min, laserScan.angle_max, self.downSampleSize)
        
    #     rospy.loginfo(str(len(self.angles)) + ' ' + str(len(laserScan.ranges)))
    #     return scipy.interpolate.splev(self.compressed_angles, scipy.interpolate.splrep(self.angles, np.array(laserScan.ranges)))
        

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
