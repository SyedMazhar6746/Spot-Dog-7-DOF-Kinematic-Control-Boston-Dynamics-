#include <iostream>
#include <vector>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/JointState.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <tf/transformations.h>
#include "arm_kinematics_inv.hpp"
#include "class_define.hpp" 


class RobotModel {
private:
    std::vector<bool> revolute;
    int dof;
    std::vector<double> goal;
    std::vector<double> theta;
    std::vector<double> dq;
    Manipulator robot;
    std::vector<double> q;
    std::vector<double> err_list;
    std::vector<double> tt;
    int ttt;
    std::vector<std::vector<double>> joint_limits;
    int stop;
    ros::Publisher pose_EE_pub;
    ros::Publisher goal_check;
    ros::Publisher joint_velocity;
    ros::Subscriber joints_sub;
    ros::Subscriber goal_sub;
    std::vector<Task*> tasks;

public:
    RobotModel() : revolute({true, true, true, true, true, true, true}), dof(revolute.size()), goal(nullptr),
                   theta(dof, 0.0), dq(dof, 0.0), robot(theta, revolute), q(dof, 0.0), err_list(), tt(), ttt(0),
                   joint_limits(6, std::vector<double>(2, 0.0)), stop(0) {
        // Initialize ROS node
        ros::NodeHandle nh;

        // Publishers
        pose_EE_pub = nh.advertise<geometry_msgs::PoseStamped>("/pose_EE", 10);
        goal_check = nh.advertise<geometry_msgs::PoseStamped>("/goal_check", 10);
        joint_velocity = nh.advertise<trajectory_msgs::JointTrajectory>("/spot_arm/joint_group_controller/command", 10);

        // Subscribers
        joints_sub = nh.subscribe("spot_arm/joint_states", 10, &RobotModel::jointStateCallback, this);
        goal_sub = nh.subscribe("/goal_set", 10, &RobotModel::goalsSetCallback, this);

        // Initial joint positions
        q[1] = -2.9; 
        q[2] = 3.0;

        // Send initial velocity command
        sendVelocity(q);
    }

    void goalsSetCallback(const std_msgs::Float64MultiArray::ConstPtr& goal_msg) {
        // Set desired goal
        goal = {goal_msg->data[0], goal_msg->data[1], goal_msg->data[2], goal_msg->data[3], goal_msg->data[4], goal_msg->data[5]};
        // Publish goal check
        geometry_msgs::PoseStamped g_pose = goalPose({goal[0], goal[1], goal[2]}, {goal[3], goal[4], goal[5]});
        goal_check.publish(g_pose);

        // Define joint limits (usually first three joints)
        joint_limits = {
            {3.0, -2.5},
            {0.42, -3.0},
            {3.0, 0.0},
            {3.0, -2.5},
            {1.5, -1.5},
            {2.7, -2.7}
        };

        // Initialize tasks
        tasks = {
            new Configuration3D("End-effector configuration", {goal[0], goal[1], goal[2], goal[5]}, dof),
            new JointPosition("first joint position", {0.0, 0.0, 0.0}, 1, joint_limits[0]),
            new JointPosition("second joint position", {0.0, 0.0, 0.0}, 2, joint_limits[1]),
            new JointPosition("third joint position", {0.0, 0.0, 0.0}, 3, joint_limits[2]),
            new JointPosition("fourth joint position", {0.0, 0.0, 0.0}, 4, joint_limits[3]),
            new JointPosition("fifth joint position", {0.0, 0.0, 0.0}, 5, joint_limits[4]),
            new JointPosition("sixth joint position", {0.0, 0.0, 0.0}, 6, joint_limits[5]),
            new Position3D("End-effector position", {goal[0], goal[1], goal[2]}, dof),
            new Orientation2D("first joint position", {0.0, 0.0, 0.0})
        };
    }

    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& data) {
        if (!goal.empty()) {
            // Update joint positions
            for (size_t i = 0; i < dof; ++i) {
                theta[i] = data->position[i];
            }

            double dt = 0.01;
            std::vector<std::vector<double>> P(dof, std::vector<double>(dof, 0.0));

            for (size_t i = 0; i < tasks.size(); ++i) {
                tasks[i]->update(robot);

                if (tasks[i]->boolIsActive()) {
                    std::vector<double> err = tasks[i]->getError();
                    std::vector<std::vector<double>> J = tasks[i]->getJacobian();
                    std::vector<std::vector<double>> J_bar = matrixMultiply(J, P);
                    std::vector<std::vector<double>> J_DLS = W_DLS(J_bar, 0.1);
                    std::vector<std::vector<double>> J_pinv = pseudoInverse(J_bar);

                    std::vector<std::vector<double>> dq_updated = matrixMultiply(J_DLS, matrixSubtract(err, matrixMultiply(J, dq)));
                    dq = matrixAdd(dq, dq_updated);

                    P = matrixSubtract(P, matrixMultiply(J_pinv, J_bar));
                }
            }

            double p_v = 0.1;
            double n_v = -0.1;

            for (size_t m = 0; m < dq.size(); ++m) {
                if (dq[m][0] < n_v) {
                    dq = scaleVector(dq, n_v, m);
                }
                if (dq[m][0] > p_v) {
                    dq = scaleVector(dq, p_v, m);
                }
            }

            q = matrixAdd(q, matrixMultiply(dq, dt));
            sendVelocity(q);
            robot.update(dq, dt);

            // Publishing end-effector pose
            geometry_msgs::PoseStamped pose_eef = poseEE(armKinematics(q));
            pose_EE_pub.publish(pose_eef);
        }
    }

    void sendVelocity(const std::vector<double>& q) {
        size_t num_joints = q.size();
        trajectory_msgs::JointTrajectory joint_traj;
        joint_traj.header.stamp = ros::Time::now();
        joint_traj.joint_names = {"spot_arm/joint_" + std::to_string(i + 1) for (size_t i = 0; i < num_joints; ++i)};

        trajectory_msgs::JointTrajectoryPoint point;
        point.positions = q;
        point.time_from_start = ros::Duration(1.0);
        joint_traj.points.push_back(point);
        joint_velocity.publish(joint_traj);
        ++stop;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "kinematics_node");
    RobotModel model;
    ros::spin();
    return 0;
}