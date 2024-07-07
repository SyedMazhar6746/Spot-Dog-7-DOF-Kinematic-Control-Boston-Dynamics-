#include <iostream>
#include <vector>
#include <cmath>
#include "arm_kinematics_inv.hpp" 

class Manipulator {
private:
    std::vector<double> theta;
    std::vector<bool> revolute;
    int dof;
    std::vector<std::vector<double>> T;

public:
    Manipulator(const std::vector<double>& theta, const std::vector<bool>& revolute) 
        : theta(theta), revolute(revolute), dof(revolute.size()) {}

    void update(const std::vector<double>& dq, double dt) {
        for (int i = 0; i < dof; ++i) {
            if (revolute[i]) {
                theta[i] += dq[i] * dt;
            }
        }
        T = arm_kinematics(theta);
    }

    std::vector<std::vector<double>> getEEJacobian() {
        return Jacobian(theta, dof);
    }

    std::vector<double> getEETransform() {
        T = arm_kinematics(theta);
        return T.back(); 
    }

    std::vector<std::vector<double>> get_list_Transform() {
        T = arm_kinematics(theta);
        return T; 
    }

    double getJointPos(int joint) {
        return theta[joint - 1]; 
    }

    int getDOF() {
        return dof;
    }

    std::vector<std::vector<double>> getLinkJacobian(int link) {
        return Jacobian(theta, link); 
    }

    std::vector<double> get_Se_LTransform(int link) {
        return T[link];
    }
};


class Task {
private:
    std::string name;
    std::vector<double> sigma_d;
    bool active;
    double norm_err;
    
    std::vector<std::vector<double>> J; // Task Jacobian
    std::vector<double> err; // Task error (tilde sigma)

public:
    // Constructor
    Task(const std::string& name, const std::vector<double>& desired)
        : name(name), sigma_d(desired), active(false), norm_err(200) {
    }

    // Method updating the task variables
    virtual void update(Manipulator& robot) = 0;

    // Method checking if the task is active or not
    bool bool_is_Active() {
        return active;
    }

    // Method setting the desired sigma
    void setDesired(const std::vector<double>& value) {
        sigma_d = value;
    }

    // Method returning the desired sigma
    std::vector<double> getDesired() {
        return sigma_d;
    }

    // Method returning the task Jacobian
    std::vector<std::vector<double>> getJacobian() {
        return J;
    }

    // Method returning the task error (tilde sigma)
    std::vector<double> getError() {
        return err;
    }

    // Method returning the norm of the error
    double n_err() {
        return norm_err;
    }
};



class Position3D : public Task {
private:
    int link;
    int des_dim;
    std::vector<std::vector<double>> J;
    std::vector<double> err;

public:
    Position3D(const std::string& name, const std::vector<double>& desired, int link)
        : Task(name, desired), link(link), des_dim(desired.size()), J(des_dim, std::vector<double>(link)), err(des_dim, 0.0) {
        active = true;
    }

    void update(Manipulator& robot) override {
        J = robot.getEEJacobian().submatrix(0, 3, 0, link); // Update task Jacobian

        auto X = robot.getEETransform();
        std::vector<double> eep(3);
        eep[0] = X[0][3];
        eep[1] = X[1][3];
        eep[2] = X[2][3];

        norm_err = std::sqrt(std::pow(getDesired()[0] - eep[0], 2) +
                             std::pow(getDesired()[1] - eep[1], 2) +
                             std::pow(getDesired()[2] - eep[2], 2));

        for (int i = 0; i < des_dim; ++i) {
            err[i] = getDesired()[i] - eep[i];
        }
    }
};


class Orientation2D : public Task {
private:
    int des_dim;
    std::vector<std::vector<double>> J;
    std::vector<double> err;

public:
    Orientation2D(const std::string& name, const std::vector<double>& desired)
        : Task(name, desired), des_dim(desired.size()), J(des_dim, std::vector<double>(7)), err(des_dim, 0.0) {
        active = true;
    }

    void update(Manipulator& robot) override {
        J = robot.getEEJacobian().submatrix(3, 6, 0, 7); // Update task Jacobian

        auto Tr = robot.getEETransform();
        double roll = std::atan2(Tr[1][0], Tr[0][0]);
        double pitch = std::atan2(-Tr[2][0], (Tr[0][0] * std::cos(roll) + Tr[1][0] * std::sin(roll)));
        double yaw = std::atan2((-Tr[1][2] * std::cos(roll) + Tr[0][2] * std::sin(roll)), (Tr[1][1] * std::cos(roll) + Tr[0][1] * std::sin(roll)));
        std::vector<double> orien = {roll, pitch, yaw};

        for (int i = 0; i < des_dim; ++i) {
            err[i] = getDesired()[i] - orien[i];
        }
    }
};


class Configuration3D : public Task {
private:
    int link;
    int des_dim;
    std::vector<std::vector<double>> J;
    std::vector<double> err;
    std::vector<double> config;

public:
    Configuration3D(const std::string& name, const std::vector<double>& desired, int link)
        : Task(name, desired), link(link), des_dim(desired.size()), J(des_dim, std::vector<double>(7)), err(des_dim, 0.0), config(des_dim, 0.0) {
        active = true;
        sigma_d = desired;
    }

    void update(Manipulator& robot) override {
        J = robot.getEEJacobian().submatrix(0, 3, 0, link); // Update task Jacobian

        auto eef_transf = robot.getEETransform();
        std::vector<double> eep(3);
        eep[0] = eef_transf[0][3];
        eep[1] = eef_transf[1][3];
        eep[2] = eef_transf[2][3];

        std::vector<double> q = robot.quaternion_from_matrix(eef_transf);
        std::vector<double> eu = robot.euler_from_quaternion(q);
        double orien = eu[0];

        for (int i = 0; i < des_dim; ++i) {
            if (i < 3) {
                config[i] = eep[i];
            } else {
                config[i] = orien;
            }
            err[i] = getDesired()[i] - config[i];
        }
    }
};


class JointPosition : public Task {
private:
    int link;
    int des_dim;
    std::vector<std::vector<double>> J;
    double activation[4]; 
    bool active;

public:
    JointPosition(const std::string& name, const std::vector<double>& desired, int link, const std::vector<double>& activation)
        : Task(name, desired), link(link), des_dim(desired.size()), J(des_dim, std::vector<double>(7)), active(false) {
        std::copy(activation.begin(), activation.end(), this->activation);
    }

    void update(Manipulator& robot) override {
        J = robot.getLinkJacobian(link).submatrix(3, 6, 0, 7); // Update task Jacobian 

        double joint_pos = robot.getJointPos(link);
        if (!active && joint_pos > activation[0]) {
            active = true;
            err[0] = -10.0;
        }

        if (!active && joint_pos < activation[2]) {
            active = true;
            err[0] = 5.0;
        }

        if (active && joint_pos < activation[1] && joint_pos > activation[0]) {
            active = false;
            err[0] = 0.0;
        }

        if (active && joint_pos > activation[3] && joint_pos < activation[2]) {
            active = false;
            err[0] = 0.0;
        }
    }
};