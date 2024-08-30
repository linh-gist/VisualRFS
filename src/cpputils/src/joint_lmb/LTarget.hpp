//
// Created by Linh Ma (linh.mavan@gm.gist.ac.kr) on 21. 11. 17..
//
#ifndef JOINT_LMB_TARGET_HPP
#define JOINT_LMB_TARGET_HPP

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <Eigen/Dense>
#include "LModel.hpp"

using namespace Eigen;

// track table for GLMB (cell array of structs for individual tracks)
// (1) r: existence probability
// (2) Gaussian Mixture w (weight), m (mean), P (covariance matrix)
// (3) Label: birth time & index of target at birth time step
// (4) gatemeas: indexes gating measurement (using  Chi-squared distribution)
// (5) deep feature: representing feature information extracted from re-identification network
class LTarget {
private:
    int x_dim = 8;
    int max_cpmt = 500;

    // wg, mg, Pg, ..., store temporary Gaussian mixtures while updating, see 'update_gms'
    VectorXd wg;
    MatrixXd mg;
    std::vector<MatrixXd> Pg;
    // store number of Gaussian mixtures before updating, see 'update_gms'
    int idxg;
    float alpha_feat = 0.9;
    bool mUseFeat;

public:
    VectorXd w;
    MatrixXd m;
    std::vector<MatrixXd> P;
    VectorXd feat;
    double r;
    double r_max;
    int l;
    int last_activate;
    VectorXi gatemeas;
    int gm_len;

    LTarget(VectorXd z, VectorXd z_feat, double prob_birth, int label, bool useFeat) {
        this->wg.resize(max_cpmt);
        this->mg.resize(x_dim, max_cpmt);
        this->Pg.resize(this->max_cpmt, MatrixXd(x_dim, x_dim));
        idxg = 0;
        gm_len = 0;
        m.resize(x_dim, 1);
        m.setZero();
        m(seq(0, 3), 0) = z;
        double wh2 = (z[3] + z[2] * z[3]) / 2;  // half perimeter
        VectorXd p_init(x_dim);
        p_init << wh2, wh2, 1, wh2, wh2, wh2, 1, wh2;
        P = std::vector<MatrixXd>(1, p_init.asDiagonal());
        feat = z_feat;
        r = prob_birth;
        w.resize(1);
        w(0) = 1;
        l = label;
        r_max = 0;
        last_activate = 0;
        mUseFeat = useFeat;
    }

    void predict_gms(LModel model) {
        r = model.P_S * r;
        int plength = m.cols();
        for (int i = 0; i < plength; i++) {
            this->m.col(i) = model.F * this->m.col(i);
            this->P[i] = model.Q + model.F * (this->P[i] * model.F.transpose());
        }
    }

    VectorXd update_gms(LModel model, MatrixXd z, MatrixXd z_feat) {
        // gating by tracks, store gating index in 'gatemeas'
        gating_gms(model, z, z_feat);

        idxg = 0;
        // Gaussian mixtures for misdetection
        int length = w.size();
        wg(seq(0, length - 1)) = w;
        mg(all, seq(0, length - 1)) = m;
        for (int i = 0; i < length; i++) {
            Pg[i] = P[i];
        }
        idxg += length;
        gm_len = length;

        // Kalman update for each Gaussian with each gating measurement
        VectorXd cost_update(gatemeas.size());
        int cost_idx = 0;
        ArrayXd qz_temp(length);
        for (int emm: gatemeas) {
            // kalman_update_multiple
            VectorXd z_emm = z.row(emm);
            for (int i = 0; i < length; i++) {
                // KF update for a single Gaussian component
                VectorXd mu = model.H * m.col(i);
                MatrixXd S = model.R + ((model.H * P[i]) * model.H.transpose());
                MatrixXd Vs = S.llt().matrixL();
                double det_S = pow(Vs.diagonal().prod(), 2);
                MatrixXd inv_sqrt_S = Vs.inverse();
                MatrixXd iS = inv_sqrt_S * inv_sqrt_S.transpose();
                MatrixXd K = (P[i] * model.H.transpose()) * iS;
                VectorXd z_mu = z_emm - mu;

                qz_temp(i) = exp(-0.5 * (z_emm.size() * log(2 * M_PI) + log(det_S) + (z_mu.transpose() * (iS * z_mu))));
                mg.col(idxg + i) = m.col(i) + (K * z_mu);
                Pg[idxg + i] = (MatrixXd::Identity(x_dim, x_dim) - (K * model.H)) * P[i];
            }

            VectorXd w_temp = (qz_temp * w.array()) + std::nexttoward(0.0, 1.0L);
            double pm_temp = 1;
            if (mUseFeat) {
                double cdist = ((VectorXd) z_feat.row(emm) - this->feat).norm();
                pm_temp = 0.1 * pow(cdist, 14) + 0.9 * pow((2 - cdist), 14);
            }
            cost_update[cost_idx] = w_temp.sum() * pm_temp;
            cost_idx += 1;
            wg(seq(idxg, idxg + length - 1)) = w_temp / w_temp.sum();
            idxg += length;
        }

        // Copy values back to each fields
        w = wg(seq(0, idxg - 1));
        m = mg(all, seq(0, idxg - 1));
        P.assign(Pg.begin(), Pg.begin() + idxg);

        return cost_update;
    }

    void gating_gms(LModel model, MatrixXd z, MatrixXd z_feat) {
        int zlength = z.rows();
        int plength = this->m.cols();
        if (zlength == 0) {
            return;
        }
        gatemeas = VectorXi::Ones(zlength);
        gatemeas *= -1;
        VectorXi cdist_gate = VectorXi::Zero(zlength);
        if (mUseFeat) {
            VectorXd feat_norm = (z_feat.transpose().colwise().norm() * this->feat.norm());
            VectorXd cdist = (z_feat * this->feat).array() / feat_norm.array();
            cdist_gate = ((1 - cdist.array()) < 0.3).cast<int>(); //cosine distance less than 0.3
        }
        for (int j = 0; j < plength; j++) {
            MatrixXd Sj = model.R + ((model.H * P[j]) * model.H.transpose());
            MatrixXd Vs = Sj.llt().matrixL();
            MatrixXd inv_sqrt_Sj = Vs.inverse();
            MatrixXd nu = z.transpose().colwise() - model.H * m.col(j);
            MatrixXd square_Sjnu = (inv_sqrt_Sj * nu).array().square();
            VectorXd dist = square_Sjnu.colwise().sum();
            VectorXi dist_gate = (dist.array() < model.gamma).cast<int>();
            for (int i = 0; i < zlength; i++) {
                if (dist_gate(i) || cdist_gate(i)) {
                    gatemeas(i) = i;
                }
            }
        }
        std::vector<int> result;
        for (int i = 0; i < zlength; i++) {
            if (gatemeas(i) >= 0) {
                result.push_back(i);
            }
        }
        gatemeas = VectorXi::Map(result.data(), result.size());
    }

    void re_activate(VectorXd z, VectorXd z_feat, double prob_birth) {
        this->m.resize(x_dim, 1);
        this->m.setZero();
        this->m(seq(0, 3), 0) = z;
        VectorXd p_init(x_dim);
        double wh2 = (z[3] + z[2] * z[3]) / 2;  // half perimeter
        p_init << wh2, wh2, 1, wh2, wh2, wh2, 1, wh2;
        this->P = std::vector<MatrixXd>(1, p_init.asDiagonal());
        this->r = prob_birth;
        w.resize(1);
        w(0) = 1;
        this->feat = z_feat;
    }

    void select_gms(VectorXi select_idxs) {
        w = wg(select_idxs);
        m = mg(all, select_idxs);
        std::vector<MatrixXd> selectedP;
        for (int i: select_idxs) {
            selectedP.push_back(Pg[i]);
        }
        P = selectedP;
    }

    void finalize_glmb2lmb(VectorXd sums, int association_idx, MatrixXd z_feat, int time_step) {
        if (association_idx > 0) {   // association_idx = 0, misdetection, keeping the same feature
            if (mUseFeat) {
                VectorXd association_feat = z_feat.row(association_idx - 1);
                feat = alpha_feat * feat + (1 - alpha_feat) * association_feat;
                feat /= feat.norm();
            }
            last_activate = time_step;  // only update if the highest hypothesis weight is not miss-detection
        }
        VectorXd repeat_sums = sums(ArrayXi::LinSpaced(sums.size() * gm_len, 0, sums.size() - 1));
        w = w.array() * repeat_sums.array();
        r = w.sum();
        if (r_max < r) {
            r_max = r;
        }
        w = w / r;
    }

    void cleanup(double elim_threshold = 1e-5, int l_max = 10) {
        // Gaussian prune, remove components that have weight lower than a threshold
        int select_idx = 0;
        for (int i = 0; i < w.size(); i++) {
            if (w(i) > elim_threshold) {
                w(select_idx) = w(i);
                m.col(select_idx) = m.col(i);
                P[select_idx] = P[i];
                select_idx += 1;
            }
        }
        VectorXd w_temp = w(seq(0, select_idx - 1));
        MatrixXd m_temp = m(all, seq(0, select_idx - 1));
        w = w_temp;
        m = m_temp;
        P.erase(P.begin() + select_idx, P.end());
        // Gaussian cap, limit on number of Gaussians in each track
        if (w.size() > l_max) {
            std::vector<double> v_w(w.size());
            VectorXd::Map(&v_w[0], w.size()) = w;
            std::vector<int> idx(w.size());
            std::iota(idx.begin(), idx.end(), 0);
            stable_sort(idx.begin(), idx.end(), [&v_w](int i1, int i2) { return v_w[i1] > v_w[i2]; });
            VectorXi idx_eigen = VectorXi::Map(idx.data(), idx.size());
            VectorXi idxkeep_eigen = idx_eigen(seq(0, l_max - 1));

            VectorXd w_new = w(idxkeep_eigen);
            w_temp = w_new * (w.sum() / w_new.sum());
            m_temp = m(all, idxkeep_eigen);
            w = w_temp;
            m = m_temp;
            std::vector<MatrixXd> newP(l_max);
            for (int i = 0; i < l_max; i++) {
                newP[i] = P[idx[i]];
            }
            P = newP;
        }
    }
};

#endif //JOINT_LMB_TARGET_HPP