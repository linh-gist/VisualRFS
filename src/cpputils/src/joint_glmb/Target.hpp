//
// Created by Linh Ma (linh.mavan@gm.gist.ac.kr) on 21. 11. 17..
//
#ifndef JOINT_GLMB_TARGET_HPP
#define JOINT_GLMB_TARGET_HPP

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <Eigen/Dense>
#include "Model.hpp"

using namespace Eigen;

class Target {
private:
    float alpha_feat = 0.9;
    float wh2 = 50;
    MatrixXd P;
    int x_dim = 8;
    bool mUseFeat;
public:
    VectorXd m;
    VectorXd feat;
    double r;
    int l;
    int last_activate;
    double P_S;
    double region[2];
    int birth_time;
    VectorXi gatemeas;

    Target() {};

    Target(VectorXd z, VectorXd feat, double prob_birth, int birth_time, int label, float region[2], bool useFeat) {
        this->m = VectorXd::Zero(z.size() * 2);
        this->m(seq(0, 3)) = z;
        VectorXd p_init(x_dim);
        p_init << wh2, wh2, 1, wh2, wh2, wh2, 1, wh2;
        this->P = p_init.asDiagonal();
        this->feat = feat;
        this->r = prob_birth;
        this->l = label;
        this->last_activate = 0;
        this->P_S = 0.99;
        this->region[0] = region[0];
        this->region[1] = region[1];
        this->birth_time = birth_time;
        mUseFeat = useFeat;
    }

    void predict(Model model, int time_step, double average_area) {
        this->m = model.F * this->m;
        this->P = model.Q + model.F * (this->P * model.F.transpose());
        double w = std::max(0.2, m[2]) * m[3];
        double t = m[1] - m[3] / 2;
        double l = m[0] - w / 2;
        double b = m[1] + m[3] / 2;
        double r = m[0] + w / 2;
        if ((r > 0) && (b > 0) && (t < region[1]) && (l < region[0])) {
            // unknown scene mask, b(x) = 0.99 for target inside image
            // A labeled random finite set online multi-object tracker for video data, Du Yong Kim, eq(11)
            P_S = 0.95 / (1 + exp(0.75 * (birth_time - time_step)));
            P_S = P_S / (1 + exp(3.5 * (0.1 - w * m[3] / average_area)));
        } else {
            // lower survival probability for target outside of image [0, 0, width, height]
            P_S = 0.1;
        }
    }

    void update(Model model, VectorXd z, VectorXd z_feat, int time_step, Target &t, double &cost) {
        last_activate = time_step;
        // deep copy this Target to a new Target
        t.r = this->r;
        t.l = this->l;
        t.last_activate = time_step;
        t.P_S = this->P_S;
        t.region[0] = this->region[0];
        t.region[1] = this->region[1];
        t.birth_time = this->birth_time;
        t.feat = this->feat;
        t.mUseFeat = this->mUseFeat;

        // KF update
        VectorXd mu = model.H * m;
        MatrixXd S = model.R + ((model.H * P) * model.H.transpose());
        MatrixXd Vs = S.llt().matrixL();
        double det_S = pow(Vs.diagonal().prod(), 2);
        MatrixXd inv_sqrt_S = Vs.inverse();
        MatrixXd iS = inv_sqrt_S * inv_sqrt_S.transpose();
        MatrixXd K = (P * model.H.transpose()) * iS;
        VectorXd z_mu = z - mu;
        double qz_temp = exp(-0.5 * (z.size() * log(2 * M_PI) + log(det_S) + (z_mu.transpose() * (iS * z_mu))));
        t.m = m + (K * z_mu);
        t.P = (MatrixXd::Identity(P.rows(), P.rows()) - (K * model.H)) * P;

        double pm_temp_sum = 1;
        if (mUseFeat) {
            double cdist = (z_feat - this->feat).norm();
            VectorXd feat_temp = alpha_feat * this->feat + (1 - alpha_feat) * z_feat;
            t.feat = feat_temp / feat_temp.norm();
            pm_temp_sum = 0.01 * pow(cdist, 15) + 0.99 * pow(2 - cdist, 15);
        }

        cost = (qz_temp + std::nexttoward(0.0, 1.0L)) * pm_temp_sum;
    }

    void gating(Model model, MatrixXd z, MatrixXd z_feat) {
        int zlength = z.rows();
        if (zlength == 0) {
            return;
        }
        gatemeas = VectorXi::Ones(zlength);
        gatemeas *= -1;
        MatrixXd Sj = model.R + ((model.H * P) * model.H.transpose());
        MatrixXd Vs = Sj.llt().matrixL();
        MatrixXd inv_sqrt_Sj = Vs.inverse();
        MatrixXd nu = z.transpose().colwise() - model.H * m;
        MatrixXd square_Sjnu = (inv_sqrt_Sj * nu).array().square();
        VectorXd dist = square_Sjnu.colwise().sum();
        VectorXi dist_gate = (dist.array() < model.gamma).cast<int>();

        VectorXi cdist_gate = VectorXi::Zero(zlength);
        if (mUseFeat) {
            VectorXd feat_norm = (z_feat.transpose().colwise().norm() * this->feat.norm());
            VectorXd cdist = (z_feat * this->feat).array() / feat_norm.array();
            cdist_gate = ((1 - cdist.array()) < 0.3).cast<int>(); //cosine distance less than 0.3
        }
        for (int i = 0; i < zlength; i++) {
            if (dist_gate(i) || cdist_gate(i)) {
                gatemeas(i) = i;
            }
        }
    }

    void re_activate(VectorXd z, VectorXd z_feat, double prob_birth, int time_step) {
        this->m.setZero();
        this->m(seq(0, 3)) = z;
        VectorXd p_init(x_dim);
        p_init << wh2, wh2, 1, wh2, wh2, wh2, 1, wh2;
        this->P = p_init.asDiagonal();
        this->r = prob_birth;
        this->feat = z_feat;
        this->last_activate = time_step;
    }
};

#endif //JOINT_GLMB_TARGET_HPP