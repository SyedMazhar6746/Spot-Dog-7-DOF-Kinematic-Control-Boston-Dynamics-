#include "arm_kinematics_inv.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>


// Utility functions to replace sin and cos
inline double s(double angle) { return std::sin(angle); }
inline double c(double angle) { return std::cos(angle); }

// Function provides rotation along one axis as transformation matrix 
Eigen::Matrix4d rot(char axis, double theta) {
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();

    Eigen::AngleAxisd rotation;
    if (axis == 'x') {
        rotation = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitX());
    } else if (axis == 'y') {
        rotation = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY());
    } else if (axis == 'z') {
        rotation = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
    } else {
        throw std::invalid_argument("Invalid axis. Must be 'x', 'y', or 'z'.");
    }
    
    matrix.block<3,3>(0,0) = rotation.toRotationMatrix();
    return matrix;
}

// Function provides translation as transformation matrix
Eigen::Matrix4d transl(const Eigen::Vector3d& translation) {
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    matrix.block<3,1>(0,3) = translation;
    return matrix;
}

// Function to evaluate the kinematics of the robot
std::vector<Eigen::Matrix4d> arm_kinematics(const Eigen::Matrix<double, 7, 1>& theta_list) {
    std::vector<Eigen::Matrix4d> T;
    T.push_back(Eigen::Matrix4d::Identity());

    Eigen::Matrix<double, 7, 1> theta = theta_list;

    Eigen::Matrix4d T1 = transl(Eigen::Vector3d(0.0, 0, 0.357)) * rot('z', theta(0));
    Eigen::Matrix4d T2 = rot('y', theta(1)) * transl(Eigen::Vector3d(0.3385, 0, 0));
    Eigen::Matrix4d T3 = rot('y', theta(2));
    Eigen::Matrix4d T4 = transl(Eigen::Vector3d(0.140, 0, 0.0750)) * rot('x', theta(3));
    Eigen::Matrix4d T5 = transl(Eigen::Vector3d(0.2633, 0, 0.0)) * rot('y', theta(4));
    Eigen::Matrix4d T6 = rot('x', theta(5));
    Eigen::Matrix4d T7 = transl(Eigen::Vector3d(0.11745, 0, 0.01482));

    std::vector<Eigen::Matrix4d> T_list = {T1, T2, T3, T4, T5, T6, T7};
    
    for (const auto& t : T_list) {
        T.push_back(T.back() * t);
    }
    return T;
}

// Function to publish the end-effector pose
geometry_msgs::PoseStamped pose_EE(const Eigen::Matrix4d& transformation_matrix) {
    Eigen::Vector3d t = transformation_matrix.block<3,1>(0,3);
    tf2::Quaternion q;
    tf2::Matrix3x3 tf_matrix;
    tf_matrix.setValue(
        transformation_matrix(0, 0), transformation_matrix(0, 1), transformation_matrix(0, 2),
        transformation_matrix(1, 0), transformation_matrix(1, 1), transformation_matrix(1, 2),
        transformation_matrix(2, 0), transformation_matrix(2, 1), transformation_matrix(2, 2)
    );
    tf_matrix.getRotation(q);

    geometry_msgs::PoseStamped p;
    p.header.frame_id = "base_link";
    p.header.stamp = ros::Time::now();

    p.pose.position.x = t.x();
    p.pose.position.y = t.y();
    p.pose.position.z = t.z();

    p.pose.orientation.x = q.x();
    p.pose.orientation.y = q.y();
    p.pose.orientation.z = q.z();
    p.pose.orientation.w = q.w();

    return p;
}

// Function to publish the goal pose
geometry_msgs::PoseStamped goal_pose(const Eigen::Vector3d& trans, const Eigen::Vector3d& angle) {
    tf2::Quaternion q;
    q.setRPY(angle(0), angle(1), angle(2));

    geometry_msgs::PoseStamped g_pose;
    g_pose.header.frame_id = "base_link";
    g_pose.pose.position.x = trans.x();
    g_pose.pose.position.y = trans.y();
    g_pose.pose.position.z = trans.z();

    g_pose.pose.orientation.x = q.x();
    g_pose.pose.orientation.y = q.y();
    g_pose.pose.orientation.z = q.z();
    g_pose.pose.orientation.w = q.w();

    return g_pose;
}

// Function to find the weighted DLS (i.e. to find jacobian inverse)
Eigen::MatrixXd W_DLS(const Eigen::MatrixXd& A, double damping) {
    int cols = A.cols();
    int rows = A.rows();

    Eigen::MatrixXd w = Eigen::MatrixXd::Identity(cols, cols); // Adjust size as needed
    Eigen::MatrixXd w_i = w.inverse();

    Eigen::MatrixXd A_damped = A * w_i * A.transpose() + damping * damping * Eigen::MatrixXd::Identity(rows, rows);
    Eigen::MatrixXd A_damped_inv = A_damped.inverse();
    Eigen::MatrixXd A_DLS = w_i * A.transpose() * A_damped_inv;

    return A_DLS;
}

Eigen::Matrix<double, 7, 1> scale(const Eigen::Matrix<double, 7, 1>& dq, double x, int j) {
    Eigen::Matrix<double, 7, 1> scaled_dq;
    for (int i = 0; i < dq.size(); ++i) {
        if (i != j) {
            scaled_dq(i) = (x * dq(i)) / dq(j);
        } else {
            scaled_dq(i) = x;
        }
    }
    return scaled_dq;
}

Eigen::MatrixXd joint_limits(const Eigen::MatrixXd& all_joints, bool original) {
    Eigen::MatrixXd joint_limits;
    if (original) {
        Eigen::VectorXd q_max(6);
        q_max << 3.0, 0.42, 3.0, 3.0, 1.5, 2.7;
        Eigen::VectorXd q_min(6);
        q_min << -2.5, -3.0, 0.0, -2.5, -1.5, -2.7;
        joint_limits.resize(6, 4);
        joint_limits << q_max - 0.05, q_max, q_min + 0.05, q_min;
    } else {
        Eigen::MatrixXd q_max = all_joints.leftCols(1);
        Eigen::MatrixXd q_min = all_joints.rightCols(1);
        joint_limits.resize(q_max.rows(), 4);
        joint_limits << q_max - 0.05, q_max, q_min + 0.05, q_min;
    }
    return joint_limits;
} 



std::vector<std::vector<double>> Jacobian(const std::vector<double>& theta, int link) {
    // Initialize the Jacobian matrix
    std::vector<std::vector<double>> J(6, std::vector<double>(4, 0.0));

    double j11 = 0.01482 * (((-s(theta[0]) * s(theta[1]) * c(theta[2]) - s(theta[0]) * s(theta[2]) * c(theta[1])) * c(theta[3]) + s(theta[3]) * c(theta[0])) * c(theta[4]) + (s(theta[0]) * s(theta[1]) * s(theta[2]) - s(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[4])) * c(theta[5]) - 0.01482 * ((-s(theta[0]) * s(theta[1]) * c(theta[2]) - s(theta[0]) * s(theta[2]) * c(theta[1])) * s(theta[3]) - c(theta[0]) * c(theta[3])) * s(theta[5]) - 0.11745 * ((-s(theta[0]) * s(theta[1]) * c(theta[2]) - s(theta[0]) * s(theta[2]) * c(theta[1])) * c(theta[3]) + s(theta[3]) * c(theta[0])) * s(theta[4]) + 0.11745 * (s(theta[0]) * s(theta[1]) * s(theta[2]) - s(theta[0]) * c(theta[1]) * c(theta[2])) * c(theta[4]) + 0.4033 * s(theta[0]) * s(theta[1]) * s(theta[2]) - 0.075 * s(theta[0]) * s(theta[1]) * c(theta[2]) - 0.075 * s(theta[0]) * s(theta[2]) * c(theta[1]) - 0.4033 * s(theta[0]) * c(theta[1]) * c(theta[2]) - 0.3385 * s(theta[0]) * c(theta[1]);

    double j21 = 0.01482 * (((s(theta[1]) * c(theta[0]) * c(theta[2]) + s(theta[2]) * c(theta[0]) * c(theta[1])) * c(theta[3]) + s(theta[0]) * s(theta[3])) * c(theta[4]) + (-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[4])) * c(theta[5]) - 0.01482 * ((s(theta[1]) * c(theta[0]) * c(theta[2]) + s(theta[2]) * c(theta[0]) * c(theta[1])) * s(theta[3]) - s(theta[0]) * c(theta[3])) * s(theta[5]) - 0.11745 * ((s(theta[1]) * c(theta[0]) * c(theta[2]) + s(theta[2]) * c(theta[0]) * c(theta[1])) * c(theta[3]) + s(theta[0]) * s(theta[3])) * s(theta[4]) + 0.11745 * (-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * c(theta[4]) - 0.4033 * s(theta[1]) * s(theta[2]) * c(theta[0]) + 0.075 * s(theta[1]) * c(theta[0]) * c(theta[2]) + 0.075 * s(theta[2]) * c(theta[0]) * c(theta[1]) + 0.4033 * c(theta[0]) * c(theta[1]) * c(theta[2]) + 0.3385 * c(theta[0]) * c(theta[1]);

    double j31 = 0;
    double j41 = 0;
    double j51 = 0;
    double j61 = 1;

    double j12 = 0.01482 * ((-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * c(theta[3]) * c(theta[4]) + (-s(theta[1]) * c(theta[0]) * c(theta[2]) - s(theta[2]) * c(theta[0]) * c(theta[1])) * s(theta[4])) * c(theta[5]) - 0.01482 * (-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[3]) * s(theta[5]) - 0.11745 * (-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[4]) * c(theta[3]) + 0.11745 * (-s(theta[1]) * c(theta[0]) * c(theta[2]) - s(theta[2]) * c(theta[0]) * c(theta[1])) * c(theta[4]) - 0.075 * s(theta[1]) * s(theta[2]) * c(theta[0]) - 0.4033 * s(theta[1]) * c(theta[0]) * c(theta[2]) - 0.3385 * s(theta[1]) * c(theta[0]) - 0.4033 * s(theta[2]) * c(theta[0]) * c(theta[1]) + 0.075 * c(theta[0]) * c(theta[1]) * c(theta[2]);

    double j22 = 0.01482 * ((-s(theta[0]) * s(theta[1]) * s(theta[2]) + s(theta[0]) * c(theta[1]) * c(theta[2])) * c(theta[3]) * c(theta[4]) + (-s(theta[0]) * s(theta[1]) * c(theta[2]) - s(theta[0]) * s(theta[2]) * c(theta[1])) * s(theta[4])) * c(theta[5]) - 0.01482 * (-s(theta[0]) * s(theta[1]) * s(theta[2]) + s(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[3]) * s(theta[5]) - 0.11745 * (-s(theta[0]) * s(theta[1]) * s(theta[2]) + s(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[4]) * c(theta[3]) + 0.11745 * (-s(theta[0]) * s(theta[1]) * c(theta[2]) - s(theta[0]) * s(theta[2]) * c(theta[1])) * c(theta[4]) - 0.075 * s(theta[0]) * s(theta[1]) * s(theta[2]) - 0.4033 * s(theta[0]) * s(theta[1]) * c(theta[2]) - 0.3385 * s(theta[0]) * s(theta[1]) - 0.4033 * s(theta[0]) * s(theta[2]) * c(theta[1]) + 0.075 * s(theta[0]) * c(theta[1]) * c(theta[2]);

    double j32 = 0.01482 * ((s(theta[1]) * s(theta[2]) - c(theta[1]) * c(theta[2])) * s(theta[4]) + (-s(theta[1]) * c(theta[2]) - s(theta[2]) * c(theta[1])) * c(theta[3]) * c(theta[4])) * c(theta[5]) + 0.11745 * (s(theta[1]) * s(theta[2]) - c(theta[1]) * c(theta[2])) * c(theta[4]) - 0.01482 * (-s(theta[1]) * c(theta[2]) - s(theta[2]) * c(theta[1])) * s(theta[3]) * s(theta[5]) - 0.11745 * (-s(theta[1]) * c(theta[2]) - s(theta[2]) * c(theta[1])) * s(theta[4]) * c(theta[3]) + 0.4033 * s(theta[1]) * s(theta[2]) - 0.075 * s(theta[1]) * c(theta[2]) - 0.075 * s(theta[2]) * c(theta[1]) - 0.4033 * c(theta[1]) * c(theta[2]) - 0.3385 * c(theta[1]);

    double j42 = 0;
    double j52 = 1;
    double j62 = 0;

    double j13 = 0.01482 * ((-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * c(theta[3]) * c(theta[4])
                + (-s(theta[1]) * c(theta[0]) * c(theta[2]) - s(theta[2]) * c(theta[0]) * c(theta[1])) * s(theta[4])) * c(theta[5])
                - 0.01482 * (-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[3]) * s(theta[5])
                - 0.11745 * (-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[4]) * c(theta[3])
                + 0.11745 * (-s(theta[1]) * c(theta[0]) * c(theta[2]) - s(theta[2]) * c(theta[0]) * c(theta[1])) * c(theta[4])
                - 0.075 * s(theta[1]) * s(theta[2]) * c(theta[0])
                - 0.4033 * s(theta[1]) * c(theta[0]) * c(theta[2])
                - 0.4033 * s(theta[2]) * c(theta[0]) * c(theta[1])
                + 0.075 * c(theta[0]) * c(theta[1]) * c(theta[2]);

    double j23 = 0.01482 * ((-s(theta[0]) * s(theta[1]) * s(theta[2]) + s(theta[0]) * c(theta[1]) * c(theta[2])) * c(theta[3]) * c(theta[4])
                + (-s(theta[0]) * s(theta[1]) * c(theta[2]) - s(theta[0]) * s(theta[2]) * c(theta[1])) * s(theta[4])) * c(theta[5])
                - 0.01482 * (-s(theta[0]) * s(theta[1]) * s(theta[2]) + s(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[3]) * s(theta[5])
                - 0.11745 * (-s(theta[0]) * s(theta[1]) * s(theta[2]) + s(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[4]) * c(theta[3])
                + 0.11745 * (-s(theta[0]) * s(theta[1]) * c(theta[2]) - s(theta[0]) * s(theta[2]) * c(theta[1])) * c(theta[4])
                - 0.075 * s(theta[0]) * s(theta[1]) * s(theta[2])
                - 0.4033 * s(theta[0]) * s(theta[1]) * c(theta[2])
                - 0.4033 * s(theta[0]) * s(theta[2]) * c(theta[1])
                + 0.075 * s(theta[0]) * c(theta[1]) * c(theta[2]);

    double j33 = 0.01482 * ((s(theta[1]) * s(theta[2]) - c(theta[1]) * c(theta[2])) * s(theta[4])
                + (-s(theta[1]) * c(theta[2]) - s(theta[2]) * c(theta[1])) * c(theta[3]) * c(theta[4])) * c(theta[5])
                + 0.11745 * (s(theta[1]) * s(theta[2]) - c(theta[1]) * c(theta[2])) * c(theta[4])
                - 0.01482 * (-s(theta[1]) * c(theta[2]) - s(theta[2]) * c(theta[1])) * s(theta[3]) * s(theta[5])
                - 0.11745 * (-s(theta[1]) * c(theta[2]) - s(theta[2]) * c(theta[1])) * s(theta[4]) * c(theta[3])
                + 0.4033 * s(theta[1]) * s(theta[2])
                - 0.075 * s(theta[1]) * c(theta[2])
                - 0.075 * s(theta[2]) * c(theta[1])
                - 0.4033 * c(theta[1]) * c(theta[2]);
    double j43 = 0;
    double j53 = 1;
    double j63 = 0;

    double j14 = -0.11745 * (-(s(theta[1]) * c(theta[0]) * c(theta[2]) + s(theta[2]) * c(theta[0]) * c(theta[1])) * s(theta[3]) + s(theta[0]) * c(theta[3])) * s(theta[4])
                 + 0.01482 * (-(s(theta[1]) * c(theta[0]) * c(theta[2]) + s(theta[2]) * c(theta[0]) * c(theta[1])) * s(theta[3]) + s(theta[0]) * c(theta[3])) * c(theta[4]) * c(theta[5])
                 - 0.01482 * ((s(theta[1]) * c(theta[0]) * c(theta[2]) + s(theta[2]) * c(theta[0]) * c(theta[1])) * c(theta[3]) + s(theta[0]) * s(theta[3])) * s(theta[5]);

    double j24 = -0.11745 * (-(s(theta[0]) * s(theta[1]) * c(theta[2]) + s(theta[0]) * s(theta[2]) * c(theta[1])) * s(theta[3]) - c(theta[0]) * c(theta[3])) * s(theta[4])
                 + 0.01482 * (-(s(theta[0]) * s(theta[1]) * c(theta[2]) + s(theta[0]) * s(theta[2]) * c(theta[1])) * s(theta[3]) - c(theta[0]) * c(theta[3])) * c(theta[4]) * c(theta[5])
                 - 0.01482 * ((s(theta[0]) * s(theta[1]) * c(theta[2]) + s(theta[0]) * s(theta[2]) * c(theta[1])) * c(theta[3]) - s(theta[3]) * c(theta[0])) * s(theta[5]);

    double j34 = 0.11745 * (-s(theta[1]) * s(theta[2]) + c(theta[1]) * c(theta[2])) * s(theta[3]) * s(theta[4])
                 - 0.01482 * (-s(theta[1]) * s(theta[2]) + c(theta[1]) * c(theta[2])) * s(theta[3]) * c(theta[4]) * c(theta[5])
                 - 0.01482 * (-s(theta[1]) * s(theta[2]) + c(theta[1]) * c(theta[2])) * s(theta[5]) * c(theta[3]);

    double j44 = 1;
    double j54 = 0;
    double j64 = 1;

    double j15 = 0.01482 * (-((s(theta[1]) * c(theta[0]) * c(theta[2]) + s(theta[2]) * c(theta[0]) * c(theta[1])) * c(theta[3]) + s(theta[0]) * s(theta[3])) * s(theta[4])
                  + (-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * c(theta[4])) * c(theta[5])
                  - 0.11745 * ((s(theta[1]) * c(theta[0]) * c(theta[2]) + s(theta[2]) * c(theta[0]) * c(theta[1])) * c(theta[3]) + s(theta[0]) * s(theta[3])) * c(theta[4])
                  - 0.11745 * (-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[4]);

    double j25 = 0.01482 * (-((-s(theta[0]) * s(theta[1]) * c(theta[2]) + s(theta[0]) * s(theta[2]) * c(theta[1])) * c(theta[3]) - s(theta[3]) * c(theta[0])) * s(theta[4])
                  + (-s(theta[0]) * s(theta[1]) * s(theta[2]) + s(theta[0]) * c(theta[1]) * c(theta[2])) * c(theta[4])) * c(theta[5])
                  - 0.11745 * ((-s(theta[0]) * s(theta[1]) * c(theta[2]) + s(theta[0]) * s(theta[2]) * c(theta[1])) * c(theta[3]) - s(theta[3]) * c(theta[0])) * c(theta[4])
                  - 0.11745 * (-s(theta[0]) * s(theta[1]) * s(theta[2]) + s(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[4]);

    double j35 = -0.11745 * (-s(theta[1]) * s(theta[2]) + c(theta[1]) * c(theta[2])) * c(theta[3]) * c(theta[4])
                  - 0.11745 * (-s(theta[1]) * c(theta[2]) - s(theta[2]) * c(theta[1])) * s(theta[4])
                  + 0.01482 * (-(-s(theta[1]) * s(theta[2]) + c(theta[1]) * c(theta[2])) * s(theta[4]) * c(theta[3])
                  + (-s(theta[1]) * c(theta[2]) - s(theta[2]) * c(theta[1])) * c(theta[4])) * c(theta[5]);

    double j45 = 0;
    double j55 = 1;
    double j65 = 0;

    double j16 = -0.01482 * (((s(theta[1]) * c(theta[0]) * c(theta[2]) + s(theta[2]) * c(theta[0]) * c(theta[1])) * c(theta[3]) + s(theta[0]) * s(theta[3])) * c(theta[4])
                  + (-s(theta[1]) * s(theta[2]) * c(theta[0]) + c(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[4])) * s(theta[5])
                  - 0.01482 * ((s(theta[1]) * c(theta[0]) * c(theta[2]) + s(theta[2]) * c(theta[0]) * c(theta[1])) * s(theta[3]) - s(theta[0]) * c(theta[3])) * c(theta[5]);

    double j26 = -0.01482 * (((s(theta[0]) * s(theta[1]) * c(theta[2]) + s(theta[0]) * s(theta[2]) * c(theta[1])) * c(theta[3]) - s(theta[3]) * c(theta[0])) * c(theta[4])
                  + (-s(theta[0]) * s(theta[1]) * s(theta[2]) + s(theta[0]) * c(theta[1]) * c(theta[2])) * s(theta[4])) * s(theta[5])
                  - 0.01482 * ((s(theta[0]) * s(theta[1]) * c(theta[2]) + s(theta[0]) * s(theta[2]) * c(theta[1])) * s(theta[3]) + c(theta[0]) * c(theta[3])) * c(theta[5]);

    double j36 = -0.01482 * (-s(theta[1]) * s(theta[2]) + c(theta[1]) * c(theta[2])) * s(theta[3]) * c(theta[5])
                  - 0.01482 * ((-s(theta[1]) * s(theta[2]) + c(theta[1]) * c(theta[2])) * c(theta[3]) * c(theta[4])
                  + (-s(theta[1]) * c(theta[2]) - s(theta[2]) * c(theta[1])) * s(theta[4])) * s(theta[5]);

    double j46 = 1;
    double j56 = 0;
    double j66 = 1;

    double j17 = 0;
    double j27 = 0;
    double j37 = 0;
    double j47 = 0;
    double j57 = 0;
    double j67 = 0;

    array<array<double, 7>, 6> J = {{
        {j11, j12, j13, j14, j15, j16, j17},
        {j21, j22, j23, j24, j25, j26, j27},
        {j31, j32, j33, j34, j35, j36, j37},
        {j41, j42, j43, j44, j45, j46, j47},
        {j51, j52, j53, j54, j55, j56, j57},
        {j61, j62, j63, j64, j65, j66, j67}
    }};

    for (int i = 0; i < 6; ++i) {
        for (int j = link; j < 7; ++j) {
            J[i][j] = 0;
        }
    }

    return J;
}