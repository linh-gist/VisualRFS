//
// Created by Linh Ma (linh.mavan@gm.gist.ac.kr) on 21. 11. 17..
//
#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <unordered_set>
#include <unordered_set>
#include <unordered_map>
#include <omp.h>
#include <iterator>
#include "LTarget.hpp"
#include "LModel.hpp"
#include "../headers/pd.hpp"

using namespace std;

class LMB {
private:
    vector<LTarget> tt_lmb;
    vector<LTarget> tt_birth;
    VectorXd glmb_update_w;          // (2) vector of GLMB component/hypothesis weights
    LModel model;
    vector<LTarget> prune_tracks;
    MatrixXd X;
    VectorXi L;

    MatrixXd tt_lmb_xyah;            // LMB tracks state [x,y,a,h]
    MatrixXd tt_lmb_feat;            // LMB tracks reid feature
    double pd_range[2];
    lap::Murty murty;
    int id = 0;
    bool mUseFeat;
    bool mUseFuzzyPD;
    ComputePD mComputePD;
    double mAverageArea;

    void jointpredictupdate(LModel model, MatrixXd z, MatrixXd feat, int k) {
        // create birth tracks
        if (k == 0) {
            for (int idx = 0; idx < z.rows(); idx++) {
                LTarget tt = LTarget(z.row(idx), feat.row(idx), model.prob_birth, this->id, mUseFeat);
                this->id += 1;
                tt_birth.push_back(tt);
            }
            if (mUseFuzzyPD) {
                VectorXd zcol2 = z.col(2);
                VectorXd zcol3 = z.col(3);
                mAverageArea = (zcol2.array() * zcol3.array().square()).sum() / z.rows();
            }
        }
        // generate surviving tracks
        for (LTarget &tt: tt_lmb) {
            tt.predict_gms(model);
        }
        int m = z.rows();  // number of measurements
        if (m == 0) {      // see MOT16-12, frame #445
            return;  // no measurement to update, only predict existing tracks
        }
        // create predicted tracks - concatenation of birth and survival
        tt_lmb.insert(tt_lmb.end(), tt_birth.begin(), tt_birth.end());
        int ntracks = tt_lmb.size();
        tt_lmb_xyah.resize(ntracks, 4);
        tt_lmb_feat.resize(ntracks, feat.cols());
        VectorXi tt_labels(ntracks);
        for (int i = 0; i < ntracks; i++) {
            int max_gm = 0;
            tt_lmb[i].w.maxCoeff(&max_gm);
            tt_lmb_xyah.row(i) = tt_lmb[i].m.col(max_gm)(seq(0, 3));
            tt_lmb_feat.row(i) = tt_lmb[i].feat;
            tt_labels(i) = tt_lmb[i].l;
        }

        VectorXd avps(ntracks);
        VectorXd avqs(ntracks);
        VectorXd avpd(ntracks);
        VectorXd avqd(ntracks);
        // compute Intersection Over Area (between tt_lmb and estimated tracks) to find P_D for each track
        // object stands close to a camera has a higher bottom coordinate [(0, 0) : (top, left)]
        // back to front: objects from far to near a camera
        VectorXd mutual_ioi = bboxes_ioi_xyah_back2front(tt_lmb_xyah, tt_labels, X.transpose(), L);
        VectorXi active_idxs(ntracks);
        VectorXd area_rate;
        VectorXd area_all;
        if (mUseFuzzyPD) {
            VectorXd col2 = tt_lmb_xyah.col(2);
            VectorXd col3 = tt_lmb_xyah.col(3);
            area_all = col2.array() * col3.array().square();
            area_rate = area_all.array() / mAverageArea;
        }
        for (int tabidx = 0; tabidx < ntracks; tabidx++) {
            // average survival/death probabilities
            avps[tabidx] = tt_lmb[tabidx].r;
            // average detection/missed probabilities
            if (mUseFuzzyPD) {
                double area_rate_tmp = area_rate(tabidx);
                if (area_rate_tmp < 0) {
                    area_rate_tmp = 0;
                }
                if (area_rate_tmp > 2) {
                    area_rate_tmp = 2;
                }
                avpd[tabidx] = mComputePD.compute(area_rate_tmp, mutual_ioi[tabidx]);
            } else {
                avpd[tabidx] = std::min(pd_range[1], std::max(pd_range[0], mutual_ioi[tabidx]));
            }
            active_idxs(tabidx) = tt_lmb[tabidx].last_activate;
        }
        avqd = 1 - avpd.array();
        avqs = 1 - avps.array();

        // create updated tracks (single target Bayes update)
        MatrixXd allcostm = MatrixXd::Zero(ntracks, m);
        for (int tabidx = 0; tabidx < ntracks; tabidx++) {
            VectorXd cost_update = tt_lmb[tabidx].update_gms(model, z, feat);
            allcostm(tabidx, tt_lmb[tabidx].gatemeas) = cost_update;
        }
        // joint cost matrix, eta_j eq (22) "An Efficient Implementation of the GLMB"
        MatrixXd eta_j = (allcostm.array().colwise() * (avps.array() * avpd.array())) / model.model_c;
        MatrixXd jointcostm(ntracks, 2 * ntracks + m);
        VectorXd sm = (avps.array() * avqd.array()); //survived and misdetected
        jointcostm << (MatrixXd) avqs.asDiagonal(), (MatrixXd) sm.asDiagonal(), eta_j;

        // calculate best updated hypotheses/components
        // murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
        MatrixXd uasses;
        VectorXd nlcost;
        if (jointcostm.rows() > 0) {
            std::tie(uasses, nlcost) = murty.draw_solutions(-jointcostm.array().log(), model.H_upd);
            uasses = uasses.array() + 1;
        }
        // component updates
        VectorXd glmb_nextupdate_w(nlcost.size());
        MatrixXd assign_meas = MatrixXd::Zero(m, nlcost.size());  // store indexes of measurement assigned to a track
        MatrixXd ioas = bboxes_ioi_xyah_back2front_all(tt_lmb_xyah);
        active_idxs = k - active_idxs.array();
        if (mUseFuzzyPD) {
            mComputePD.set_recompute_cost(avqs, avps, allcostm, model.model_c);
        }
        for (int hidx = 0; hidx < uasses.rows(); hidx++) {
            for (int j = 0; j < uasses.cols(); j++) {
                if (uasses(hidx, j) <= ntracks) {
                    // set not born/track deaths to -inf assignment
                    uasses(hidx, j) = -std::numeric_limits<double>::infinity();
                } else if (uasses(hidx, j) > ntracks && uasses(hidx, j) <= 2 * ntracks) {
                    uasses(hidx, j) = 0; // set survived+missed to 0 assignment
                } else {
                    // set survived+detected to assignment of measurement index from 1:|Z|
                    uasses(hidx, j) -= 2 * ntracks;
                }
            }
            VectorXd update_hypcmp_tmp = uasses.row(hidx);
            double new_cost;
            if (mUseFuzzyPD) {
                new_cost = mComputePD.recompute_cost(update_hypcmp_tmp, ioas, area_all);
            } else {
                new_cost = recompute_cost(update_hypcmp_tmp, ioas, avqs, avps, allcostm, model.model_c, pd_range);
            }

            // hypothesis/component weight
            // Vo Ba-Ngu "An efficient implementation of the generalized labeled multi-Bernoulli filter." eq (20)
            double omega_z = -model.lambda_c + m * log(model.model_c) - new_cost;
            // Get measurement index from uasses (make sure minus 1 from [mindices+1])
            VectorXi meas_idx = select_nonnegative(update_hypcmp_tmp.array() - 1);
            assign_meas(meas_idx, hidx).array() = 1;
            glmb_nextupdate_w(hidx) = omega_z;
        }
        glmb_nextupdate_w = exp(glmb_nextupdate_w.array() - log_sum_exp(glmb_nextupdate_w));  // normalize weights
        VectorXd assign_prob = assign_meas * glmb_nextupdate_w; // adaptive birth weight for each measurement

        // The following implementation is optimized for GLMB to LMB (glmb2lmb)
        // Refer "The Labeled Multi-Bernoulli Filter, 2014"
        for (int i = 0; i < ntracks; i++) {
            map<int, double> u_inv;  // sorted  by key, key is measurement index
            for (int j = 0; j < nlcost.size(); j++) {
                int meas_idx = uasses(j, i);
                if (meas_idx >= 0) {
                    if (u_inv.find(meas_idx) == u_inv.end()) {
                        u_inv[meas_idx] = glmb_nextupdate_w(j);
                    } else {
                        u_inv[meas_idx] += glmb_nextupdate_w(j);
                    }
                }
            }
            int u_inv_size = u_inv.size();
            if (u_inv_size == 0) {  // no measurement association(including misdetection)
                continue;
            }
            LTarget &tt = tt_lmb[i];
            VectorXd sums(u_inv_size); // do not worry about order, u_inv map is sorted
            VectorXi select_gm(u_inv_size); // do not worry about order, u_inv map is sorted
            int association_idx = 0;
            double sums_max = 0;
            int tmp_idx = -1;
            for (auto kv: u_inv) {
                int key = kv.first;
                double value = kv.second;
                if (sums_max < value) {
                    sums_max = value;
                    association_idx = key;
                }
                // select gating measurement indexes appear in ranked assignment (u: variable)
                // 0 for mis-detection, 1->|Z| for measurement index
                tmp_idx += 1;
                sums[tmp_idx] = value;
                if (key == 0) {
                    select_gm[tmp_idx] = 0;
                    continue;
                }
                for (int j = 0; j < tt.gatemeas.size(); j++) {
                    if (tt.gatemeas(j) == key - 1) {
                        select_gm[tmp_idx] = j + 1;
                    }
                }
            }
            // select 'block of gaussian mixtures' that are in 'select_idxs'
            VectorXi l_range = VectorXi::LinSpaced(tt.gm_len, 0, tt.gm_len - 1).replicate(select_gm.size(), 1);
            VectorXi start_idx = select_gm(ArrayXi::LinSpaced(select_gm.size() * tt.gm_len, 0, select_gm.size() - 1));
            start_idx *= tt.gm_len;
            VectorXi select_idxs = l_range + start_idx;
            tt.select_gms(select_idxs);

            tt.finalize_glmb2lmb(VectorXd::Map(sums.data(), sums.size()), association_idx, feat, k);
        }
        apdative_birth(assign_prob, z, feat, model, k);
    }

    void apdative_birth(VectorXd assign_prob, MatrixXd z, MatrixXd feat, LModel model, int k) {
        tt_birth.clear();
        // make sure not_assigned_sum is not zero
        double not_assigned_sum = (1 - assign_prob.array()).sum() + std::nexttoward(0.0, 1.0L);
        for (int idx = 0; idx < assign_prob.size(); idx++) {
            if (assign_prob(idx) <= model.b_thresh) {
                // eq (75) "The Labeled Multi-Bernoulli Filter", Stephan Reuter∗, Ba-Tuong Vo, Ba-Ngu Vo, ...
                double prob_birth = min(model.prob_birth, (1 - assign_prob(idx)) / not_assigned_sum * model.lambda_b);
                prob_birth = max(prob_birth, std::nexttoward(0.0, 1.0L));  // avoid zero birth probability

                bool re_activate = re_activate_tracks(z(idx, seq(0, 3)), feat.row(idx), model, prob_birth);
                if (re_activate) {
                    continue;
                }

                LTarget tt(z.row(idx), feat.row(idx), prob_birth, id, mUseFeat);
                id += 1;
                tt_birth.push_back(tt);
            }
        }
    }

    bool re_activate_tracks(VectorXd z, VectorXd feat, LModel model, double prob_birth) {
        if (!mUseFeat) {
            return false;
        }
        if (prune_tracks.size() > 0) {
            MatrixXd prun_tt_feat(prune_tracks.size(), feat.size());
            for (int i = 0; i < prune_tracks.size(); i++) {
                prun_tt_feat.row(i) = prune_tracks[i].feat;
            }
            VectorXd feat_norm = (prun_tt_feat.transpose().colwise().norm() * feat.norm());
            VectorXd feats_dist = 1 - (prun_tt_feat * feat).array() / feat_norm.array();

            int min_idx;
            double min_feat = feats_dist.minCoeff(&min_idx);
            if (min_feat < 0.25) { // pruned track cannot update feature for few frames
                LTarget &tt = prune_tracks[min_idx];
                tt.re_activate(z, feat, prob_birth);
                tt_lmb.push_back(tt);

                prune_tracks.erase(prune_tracks.begin() + min_idx);
                return true; // recall this Target
            }
        }

        // first, checking whether new measurement overlap with existing tracks
        VectorXd ious = bbox_iou_xyah(z, tt_lmb_xyah);
        for (int i = 0; i < ious.size(); i++) {
            if (ious(i) > 0.2) { // consider as overlap with any existing tracks
                // compare re-id feature cdist with activating tracks
                double feat_dist = tt_lmb_feat.row(i) * feat;
                feat_dist /= (tt_lmb_feat.row(i).norm() * feat.norm());
                feat_dist = 1 - feat_dist;
                if (feat_dist < 0.2) {  // consider two re-identification features are similar
                    // new measurement and an existing track have similar feature, ignore this measurement
                    return true;
                }
            }
        }
        return false;
    }

    void clean_lmb(LModel model, int tim_step) {
        // prune tracks with low existence probabilities
        // extract vector of existence probabilities from LMB track table
        int ntracks = tt_lmb.size();
        VectorXd rvect(ntracks);
        vector<LTarget> tt_lmb_out;
        for (int i = 0; i < ntracks; i++) {
            rvect(i) = tt_lmb[i].r;
            if (rvect(i) > model.track_threshold) {
                tt_lmb_out.push_back(tt_lmb[i]);
            } else {
                prune_tracks.push_back(tt_lmb[i]);
            }
        }
        for (int i = prune_tracks.size() - 1; i >= 0; i--) {
            if (tim_step - prune_tracks[i].last_activate > 50) {
                prune_tracks.erase(prune_tracks.begin() + i);
            }
        }
        // cleanup tracks
        for (LTarget &tt: tt_lmb_out) {
            tt.cleanup();
        }
        tt_lmb = tt_lmb_out;
    }

    std::tuple<MatrixXd, VectorXi> extract_estimates() {
        // extract estimates via MAP cardinality and corresponding tracks
        int num_tracks = tt_lmb.size();
        VectorXd rvect(num_tracks);
        for (int i = 0; i < num_tracks; i++) {
            rvect(i) = tt_lmb[i].r;
            rvect(i) = std::min(1 - 1e-6, std::max(1e-6, rvect(i)));
        }
        // Calculate the cardinality distribution of the multi-Bernoulli RFS, prod(1-r)*esf(r/(1-r))
        ArrayXd rvect_array = rvect.array();
        VectorXd cdn = esf(rvect_array / (1 - rvect_array)).array();
        int mode = 0;
        cdn.maxCoeff(&mode);
        int N = std::min(num_tracks, mode);
        vector<int> idxcmp(num_tracks);
        iota(idxcmp.begin(), idxcmp.end(), 0);
        stable_sort(idxcmp.begin(), idxcmp.end(), [&rvect](int i1, int i2) { return rvect(i1) > rvect(i2); });
        X.resize(4, num_tracks);
        L.resize(num_tracks);
        int select_idx = 0;
        int idxtrk = 0;
        for (int n = 0; n < N; n++) {
            LTarget select_target = tt_lmb[idxcmp[n]];
            select_target.w.maxCoeff(&idxtrk);
            X.col(select_idx) = select_target.m(seq(0, 3), idxtrk);
            L(select_idx) = select_target.l;
            select_idx += 1;
        }

        // hysteresis, eq (71) "The Labeled Multi-Bernoulli Filter", Stephan Reuter∗, Ba-Tuong Vo, Ba-Ngu Vo, ...
        for (int idxx = N; idxx < num_tracks; idxx++) {
            LTarget select_target = tt_lmb[idxcmp[idxx]];
            if (select_target.r_max > 0.7 && select_target.r > 0.1) {
                select_target.w.maxCoeff(&idxtrk);
                X.col(select_idx) = select_target.m(seq(0, 3), idxtrk);
                L(select_idx) = select_target.l;
                select_idx += 1;
            }
        }
        if (select_idx > 0 && mUseFuzzyPD) {
            VectorXd xrow2 = X.row(2);
            VectorXd xrow3 = X.row(3);
            mAverageArea = (xrow2.array() * xrow3.array().square()).sum() / select_idx;
        }
        return {X(all, seq(0, select_idx - 1)), L(seq(0, select_idx - 1))};
    }

public:
    LMB(int width, int height, bool useFeat = true, bool useFuzzyPD = false) {
        glmb_update_w = VectorXd(1);
        glmb_update_w << 1;
        pd_range[0] = 0.7;
        pd_range[1] = 0.9;
        model = LModel();
        murty = lap::Murty();
        id = 0;
        mUseFeat = useFeat;
        mUseFuzzyPD = useFuzzyPD; // either use Fuzzy PD (true) or IoA PD (false)
        if (useFuzzyPD) {
            mComputePD = ComputePD("./compute_pd.fis");
        }
        mAverageArea = 1;
    }

    std::tuple<MatrixXd, VectorXi> run_lmb(MatrixXd z, MatrixXd feat, int k) {
        // tlbr to cxcyah
        z(all, seq(2, 3)) -= z(all, seq(0, 1));
        z(all, seq(0, 1)) += z(all, seq(2, 3)) / 2;
        z.col(2) = z.col(2).array() / z.col(3).array();
        // joint prediction and update
        jointpredictupdate(model, z, feat, k);
        // pruning, truncation and track cleanup
        clean_lmb(model, k);

        return extract_estimates();
    }

    std::tuple<MatrixXd, VectorXi, MatrixXd> run_lmb_feat(MatrixXd z, MatrixXd feat, int k) {
        // tlbr to cxcyah
        z(all, seq(2, 3)) -= z(all, seq(0, 1));
        z(all, seq(0, 1)) += z(all, seq(2, 3)) / 2;
        z.col(2) = z.col(2).array() / z.col(3).array();
        // joint prediction and update
        jointpredictupdate(model, z, feat, k);

        // pruning, truncation and track cleanup
        clean_lmb(model, k);

        // extract estimates via MAP cardinality and corresponding tracks
        int num_tracks = tt_lmb.size();
        VectorXd rvect(num_tracks);
        for (int i = 0; i < num_tracks; i++) {
            rvect(i) = tt_lmb[i].r;
            rvect(i) = std::min(1 - 1e-6, std::max(1e-6, rvect(i)));
        }
        ArrayXd rvect_array = rvect.array();
        VectorXd cdn = (1 - rvect_array).prod() * esf(rvect_array / (1 - rvect_array)).array();
        int mode;
        cdn.maxCoeff(&mode);
        int N = std::min(num_tracks, mode);
        vector<int> idxcmp(num_tracks);
        iota(idxcmp.begin(), idxcmp.end(), 0);
        stable_sort(idxcmp.begin(), idxcmp.end(),
                    [&rvect](int i1, int i2) { return rvect(i1) > rvect(i2); });
        X.resize(4, num_tracks);
        L.resize(num_tracks);
        MatrixXd F(feat.cols(), num_tracks);
        int select_idx = 0;
        int idxtrk = 0;
        for (int n = 0; n < N; n++) {
            LTarget select_target = tt_lmb[idxcmp[n]];
            select_target.w.maxCoeff(&idxtrk);
            X.col(select_idx) = select_target.m(seq(0, 3), idxtrk);
            L(select_idx) = select_target.l;
            F.col(select_idx) = select_target.feat;
            select_idx += 1;
        }

        // hysteresis, eq (71) "The Labeled Multi-Bernoulli Filter", Stephan Reuter∗, Ba-Tuong Vo, Ba-Ngu Vo, ...
        for (int idxx = N; idxx < num_tracks; idxx++) {
            LTarget select_target = tt_lmb[idxcmp[idxx]];
            if (select_target.r_max > 0.7 && select_target.r > 0.1) {
                select_target.w.maxCoeff(&idxtrk);
                X.col(select_idx) = select_target.m(seq(0, 3), idxtrk);
                L(select_idx) = select_target.l;
                F.col(select_idx) = select_target.feat;
                select_idx += 1;
            }
        }
        if (select_idx > 0 && mUseFuzzyPD) {
            VectorXd xrow2 = X.row(2);
            VectorXd xrow3 = X.row(3);
            mAverageArea = (xrow2.array() * xrow3.array().square()).sum() / select_idx;
        }
        return {X(all, seq(0, select_idx - 1)), L(seq(0, select_idx - 1)), F(all, seq(0, select_idx - 1))};
    }
};