#ifndef ARM_KINEMATICS_INV_H  
#define ARM_KINEMATICS_INV_H

#include <Eigen/Dense>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h> 

Eigen::Matrix4d rot(char axis, double theta);
Eigen::Matrix4d transl(const Eigen::Vector3d& translation);
std::vector<Eigen::Matrix4d> arm_kinematics(const Eigen::Matrix<double, 7, 1>& theta_list);
geometry_msgs::PoseStamped pose_EE(const Eigen::Matrix4d& transformation_matrix);
geometry_msgs::PoseStamped goal_pose(const Eigen::Vector3d& trans, const Eigen::Vector3d& angle);
Eigen::MatrixXd W_DLS(const Eigen::MatrixXd& A, double damping);
Eigen::Matrix<double, 7, 1> scale(const Eigen::Matrix<double, 7, 1>& dq, double x, int j);
Eigen::MatrixXd joint_limits(const Eigen::MatrixXd& all_joints, bool original);

#endif 
