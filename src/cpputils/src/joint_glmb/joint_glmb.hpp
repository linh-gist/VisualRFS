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
#include "Model.hpp"
#include "Target.hpp"
#include "../headers/pd.hpp"

using namespace std;

class GLMB {
private:
    vector<Target> glmb_update_tt;   // (1) track table for GLMB (individual tracks)
    VectorXd glmb_update_w;          // (2) vector of GLMB component/hypothesis weights
    vector<VectorXi> glmb_update_I;  // (3) cell of GLMB component/hypothesis labels (in track table)
    VectorXi glmb_update_n;          // (4) vector of GLMB component/hypothesis cardinalities
    VectorXd glmb_update_cdn;        // (5) cardinality distribution of GLMB
    Model model;
    vector<Target> tt_birth;
    vector<Target> tt_prune;
    MatrixXd tt_glmb_xyah;   // GLMB tracks state [x,y,a,h]
    MatrixXd tt_glmb_feat;   // GLMB tracks reid feature

    set<int> prev_tt_glmb_labels;
    vector<Target> prev_glmb_update_tt;
    vector<Target> prune_glmb_tt;
    vector<VectorXd> prune_glmb_tt_feat;
    vector<int> prune_glmb_tt_label;
    float region[2];
    double pd_range[2];
    lap::Murty murty;
    int id;
    bool mUseFeat;
    bool mUseFuzzyPD;
    ComputePD mComputePD;
    double mAverageArea;

    void jointpredictupdate(Model model, MatrixXd z, MatrixXd feat, int k) {
        // create birth tracks
        if (k == 0) {
            for (int idx = 0; idx < z.rows(); idx++) {
                Target tt = Target(z.row(idx), feat.row(idx), model.prob_birth, k, this->id, this->region, mUseFeat);
                this->id += 1;
                tt_birth.push_back(tt);
            }
            if (mUseFuzzyPD) {
                VectorXd zcol2 = z.col(2);
                VectorXd zcol3 = z.col(3);
                mAverageArea = (zcol2.array() * zcol3.array().square()).sum() / z.rows();
            }
        }
        // create surviving tracks - via time prediction (single target CK)
        for (Target &tt: glmb_update_tt) {
            tt.predict(model, k, mAverageArea);
        }
        int m = z.rows();  // number of measurements
        if (m == 0) {      // see MOT16-12, frame #445
            return;  // no measurement to update, only predict existing tracks
        }
        // create predicted tracks - concatenation of birth and survival
        vector<Target> glmb_predict_tt(tt_birth);  // copy track table back to GLMB struct
        glmb_predict_tt.insert(glmb_predict_tt.end(), glmb_update_tt.begin(), glmb_update_tt.end());

        int cpreds = glmb_predict_tt.size();
        tt_glmb_xyah.resize(cpreds, 4);
        tt_glmb_feat.resize(cpreds, feat.cols());
        for (int i = 0; i < cpreds; i++) {
            tt_glmb_xyah.row(i) = glmb_predict_tt[i].m(seq(0, 3));
            tt_glmb_feat.row(i) = glmb_predict_tt[i].feat;
        }

        // precalculation loop for average survival/death probabilities
        VectorXd avps(cpreds);
        for (int i = 0; i < tt_birth.size(); i++) { // P_S: tt_birth & glmb_update_tt
            avps(i) = glmb_predict_tt[i].r;
        }
        for (int i = tt_birth.size(); i < cpreds; i++) { // P_S: tt_birth & glmb_update_tt
            avps(i) = glmb_predict_tt[i].P_S;
        }
        VectorXd avqs = 1 - avps.array();

        // create updated tracks (single target Bayes update)
        // missed detection tracks (legacy tracks) using deepcopy
        vector<Target> tt_update(glmb_predict_tt);
        vector<Target> tt_update_new(m * cpreds);  // initialize cell array
        tt_update.insert(tt_update.end(), tt_update_new.begin(), tt_update_new.end());

        // measurement updated tracks (all pairs)
        MatrixXd allcostm = MatrixXd::Zero(cpreds, m);
        for (int tabidx = 0; tabidx < cpreds; tabidx++) {
            glmb_predict_tt[tabidx].gating(model, z, feat);
            for (int emm = 0; emm < m; emm++) {
                if (glmb_predict_tt[tabidx].gatemeas[emm] >= 0) {
                    int stoidx = cpreds * (emm + 1) + tabidx;
                    double cost_update;
                    Target tt_update_gate;
                    glmb_predict_tt[tabidx].update(model, z.row(emm), feat.row(emm), k, tt_update_gate, cost_update);

                    allcostm(tabidx, emm) = cost_update;
                    tt_update[stoidx] = tt_update_gate;
                }
            }
        }
        // joint cost matrix, eta_j is the cost matrix for survived and detected tracks
        MatrixXd eta_j = (allcostm.array().colwise() * avps.array()) / model.model_c;
        MatrixXd jointcostm(cpreds, 2 * cpreds + m);
        jointcostm << (MatrixXd) avqs.asDiagonal(), (MatrixXd) avps.asDiagonal(), eta_j;

        // gated measurement index matrix
        MatrixXi gatemeasidxs(cpreds, m);
        gatemeasidxs.setOnes();
        gatemeasidxs *= -1;
        for (int tabidx = 0; tabidx < cpreds; tabidx++) {
            gatemeasidxs(tabidx, all) = glmb_predict_tt[tabidx].gatemeas;
        }
        // component updates
        int runidx = 0;
        VectorXd glmb_nextupdate_w(model.H_upd * 2);
        vector<VectorXi> glmb_nextupdate_I(model.H_upd * 2);
        VectorXi glmb_nextupdate_n = VectorXi::Zero(model.H_upd * 2);
        // use to normalize assign_prob using glmb_nextupdate_w
        MatrixXd assign_meas = MatrixXd::Zero(m, model.H_upd * 2);
        int nbirths = tt_birth.size();

        VectorXd sqrt_hypoth_num = glmb_update_w.array().sqrt();
        VectorXi hypoth_num = (model.H_upd * sqrt_hypoth_num / sqrt_hypoth_num.sum()).cast<int>();
        MatrixXd neglog_jointcostm = -jointcostm.array().log(); // negative log cost

        //#pragma omp parallel for
        for (int pidx = 0; pidx < glmb_update_w.size(); pidx++) {
            // calculate best updated hypotheses/components
            int nexists = glmb_update_I[pidx].size();
            int ntracks = nbirths + nexists;
            // indices of all births and existing tracks  for current component
            VectorXi tindices(ntracks);
            tindices << VectorXi::LinSpaced(nbirths, 0, nbirths - 1), glmb_update_I[pidx].array() + nbirths;
            if (ntracks == 0) {
                continue;
            }
            // union indices of gated measurements for corresponding tracks
            MatrixXi gate_tindices = gatemeasidxs(tindices, all);
            std::set<int> mindices_set{gate_tindices.data(), gate_tindices.data() + gate_tindices.size()};
            std::vector<int> mindices_vec(mindices_set.begin(), mindices_set.end()); // convert set to vector
            VectorXi mindices_sorted = VectorXi::Map(mindices_vec.data(), mindices_vec.size());
            VectorXi mindices;
            if ((mindices_sorted.array() == -1).any()) { // ignore -1 value
                mindices = mindices_sorted(seq(1, mindices_sorted.size() - 1));
            } else {
                mindices = mindices_sorted;
            }
            // cost matrix - [no_birth/is_death | born/survived+missed | born/survived+detected]
            VectorXi col_mindices(2 * ntracks + mindices.size());
            col_mindices << tindices, cpreds + tindices.array(), 2 * cpreds + mindices.array();
            MatrixXd neglogcostm = neglog_jointcostm(tindices, col_mindices);

            // calculate average detection / missed probabilities
            MatrixXd glmbI_xyah(ntracks, 4);
            for (int tt_i = 0; tt_i < ntracks; tt_i++) {
                VectorXd tt_xyah = glmb_predict_tt[tindices[tt_i]].m(seq(0, 3));
                tt_xyah(2) = std::max(0.2, tt_xyah(2));
                glmbI_xyah.row(tt_i) = tt_xyah;
            }
            VectorXd avpd(ntracks);
            // object stands close to a camera has a higher bottom coordinate [(0, 0) : (top, left)]
            // back to front: objects from far to near a camera
            MatrixXd mutual_ioi = bboxes_ioi_xyah_back2front_all(glmbI_xyah);
            VectorXd overlapped_ioi;
            VectorXd area_rate;
            VectorXd area_all;
            if (mUseFuzzyPD) {
                overlapped_ioi = mutual_ioi.rowwise().maxCoeff().array();
                VectorXd col2 = glmbI_xyah.col(2);
                VectorXd col3 = glmbI_xyah.col(3);
                area_all = col2.array() * col3.array().square();
                area_rate = area_all.array() / mAverageArea;
            } else {
                overlapped_ioi = 1 - mutual_ioi.rowwise().maxCoeff().array();
            }
            for (int tabidx = 0; tabidx < ntracks; tabidx++) {
                // average detection/missed probabilities
                if (mUseFuzzyPD) {
                    double area_rate_tmp = area_rate(tabidx);
                    if (area_rate_tmp < 0) {
                        area_rate_tmp = 0;
                    }
                    if (area_rate_tmp > 2) {
                        area_rate_tmp = 2;
                    }
                    avpd[tabidx] = mComputePD.compute(area_rate_tmp, overlapped_ioi[tabidx]);
                } else {
                    avpd[tabidx] = std::min(pd_range[1], std::max(pd_range[0], overlapped_ioi(tabidx)));
                }
            }
            MatrixXd tmp = neglogcostm(all, seq(ntracks, 2 * ntracks - 1)).array().colwise() - (1 - avpd.array()).log();
            neglogcostm(all, seq(ntracks, 2 * ntracks - 1)) = tmp;
            tmp = neglogcostm(all, seq(2 * ntracks, neglogcostm.cols() - 1)).array().colwise() - avpd.array().log();
            neglogcostm(all, seq(2 * ntracks, neglogcostm.cols() - 1)) = tmp;

            // murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
            // output theta, measurement to track association
            MatrixXd uasses;
            VectorXd nlcost;
            std::tie(uasses, nlcost) = murty.draw_solutions(neglogcostm, hypoth_num[pidx]);
            uasses = uasses.array() + 1;
            if (mUseFuzzyPD) {
                mComputePD.set_recompute_cost(avqs(tindices), avps(tindices), allcostm(tindices, all), model.model_c);
            }
            for (int hidx = 0; hidx < uasses.rows(); hidx++) {
                int nextupdate_n = 0;
                for (int j = 0; j < uasses.cols(); j++) {
                    if (uasses(hidx, j) <= ntracks) {
                        // set not born/track deaths to -inf assignment
                        uasses(hidx, j) = -std::numeric_limits<double>::infinity();
                    } else if (uasses(hidx, j) > ntracks && uasses(hidx, j) <= 2 * ntracks) {
                        uasses(hidx, j) = 0; // set survived+missed to 0 assignment
                        nextupdate_n += 1;
                    } else {
                        // set survived+detected to assignment of measurement index from 1:|Z|
                        uasses(hidx, j) -= 2 * ntracks;
                        // restore original indices of gated measurements
                        uasses(hidx, j) -= 1;
                        uasses(hidx, j) = mindices((int) (uasses(hidx, j))) + 1;
                        nextupdate_n += 1;
                    }
                }
                /* generate corrresponding jointly predicted/updated hypotheses/components */
                VectorXd update_hypcmp_tmp = uasses.row(hidx);
                if (mUseFuzzyPD) {
                    nlcost[hidx] = mComputePD.recompute_cost(update_hypcmp_tmp, mutual_ioi, area_all);
                } else {
                    nlcost[hidx] = recompute_cost(update_hypcmp_tmp, mutual_ioi, avqs(tindices), avps(tindices),
                                                  allcostm(tindices, all), model.model_c, pd_range);
                }
                VectorXi glmb_update_I_tmp(ntracks);
                glmb_update_I_tmp << VectorXi::LinSpaced(nbirths, 0, nbirths - 1),
                        glmb_update_I[pidx].array() + nbirths;
                VectorXd update_hypcmp_idx = cpreds * update_hypcmp_tmp + glmb_update_I_tmp.cast<double>();
                // Get measurement index from uasses (make sure minus 1 from [mindices+1])
                VectorXi uasses_idx = select_nonnegative(update_hypcmp_tmp.array() - 1);

                // #pragma omp critical{} // only one thread can access the following code to update GLMB components
                assign_meas(uasses_idx, runidx).array() = 1;  // Setting index of measurements associate with a track
                // hypothesis/component tracks (via indices to track table)
                glmb_nextupdate_I[runidx] = select_nonnegative(update_hypcmp_idx);
                // hypothesis/component cardinality
                glmb_nextupdate_n(runidx) = nextupdate_n;
                // hypothesis/component weight eqs (20) => (15) => (17) => (4), omega_z
                // Vo Ba-Ngu "An efficient implementation of the generalized labeled multi-Bernoulli filter."
                glmb_nextupdate_w[runidx] =
                        -model.lambda_c + m * log(model.model_c) + log(glmb_update_w[pidx]) - nlcost[hidx];
                runidx += 1;
            }
        }
        glmb_nextupdate_I.resize(runidx); // or erase(glmb_nextupdate_I.begin() + runidx, glmb_nextupdate_I.end());
        VectorXd glmb_nextupdate_w_slice = glmb_nextupdate_w(seq(0, runidx - 1));

        // normalize weights
        glmb_update_w = exp(glmb_nextupdate_w_slice.array() - log_sum_exp(glmb_nextupdate_w_slice)); // 2
        // adaptive birth weight for each measurement
        VectorXd assign_prob = assign_meas(all, seq(0, runidx - 1)) * glmb_update_w;
        // create birth tracks
        apdative_birth(z, feat, assign_prob, model, k);

        // extract cardinality distribution
        glmb_update_n = glmb_nextupdate_n(seq(0, runidx - 1)); // 4
        VectorXd glmb_nextupdate_cdn(glmb_update_n.maxCoeff() + 1);
        for (int card = 0; card < glmb_nextupdate_cdn.size(); card++) {
            // extract probability of n targets
            glmb_nextupdate_cdn[card] = (glmb_update_n.array() == card).select(glmb_update_w, 0).sum();
        }
        // copy glmb update to the next time step
        glmb_update_tt = tt_update;             // 1
        glmb_update_I = glmb_nextupdate_I;      // 3
        glmb_update_cdn = glmb_nextupdate_cdn;  // 5

        // remove duplicate entries and clean track table
        clean_predict();
        clean_update(k);
    }

    void apdative_birth(MatrixXd z, MatrixXd feat, VectorXd assign_prob, Model model, int k) {
        tt_birth.clear();
        // make sure not_assigned_sum is not zero
        double not_assigned_sum = (1 - assign_prob.array()).sum() + std::nexttoward(0.0, 1.0L);
        for (int idx = 0; idx < assign_prob.size(); idx++) {
            if (assign_prob(idx) <= model.b_thresh) {
                // eq (75) "The Labeled Multi-Bernoulli Filter", Stephan Reuter∗, Ba-Tuong Vo, Ba-Ngu Vo, ...
                double prob_birth = min(model.prob_birth, (1 - assign_prob(idx)) / not_assigned_sum * model.lambda_b);
                prob_birth = max(prob_birth, std::nexttoward(0.0, 1.0L));  // avoid zero birth probability

                bool re_activate = re_activate_tracks(z(idx, seq(0, 3)), feat.row(idx), model, prob_birth, k);
                if (re_activate) {
                    continue;
                }

                Target tt(z.row(idx), feat.row(idx), prob_birth, k + 1, id, region, mUseFeat);
                id += 1;
                tt_birth.push_back(tt);
            }
        }
    }

    bool re_activate_tracks(VectorXd z, VectorXd feat, Model model, double prob_birth, int k) {
        if (!mUseFeat) {
            return false;
        }
        if (prune_glmb_tt.size() > 0) {
            MatrixXd prun_tt_feat(prune_glmb_tt_feat.size(), feat.size());
            for (int i = 0; i < prune_glmb_tt_feat.size(); i++) {
                prun_tt_feat.row(i) = prune_glmb_tt_feat[i];
            }
            VectorXd feat_norm = (prun_tt_feat.transpose().colwise().norm() * feat.norm());
            VectorXd feats_dist = 1 - (prun_tt_feat * feat).array() / feat_norm.array();

            int min_idx;
            double min_feat = feats_dist.minCoeff(&min_idx);
            if (min_feat < 0.25) { // pruned track cannot update feature for few frames
                Target tt = prune_glmb_tt[min_idx];
                tt.re_activate(z, feat, prob_birth, k);
                tt_birth.push_back(tt);

                for (int i = prune_glmb_tt_label.size() - 1; i >= 0; i--) {
                    if (prune_glmb_tt_label[i] == tt.l) {
                        prune_glmb_tt_label.erase(prune_glmb_tt_label.begin() + i);
                        prune_glmb_tt.erase(prune_glmb_tt.begin() + i);
                        prune_glmb_tt_feat.erase(prune_glmb_tt_feat.begin() + i);
                    }
                }
                return true; // recall this Target
            }
        }
        // first, checking whether new measurement overlap with existing tracks
        VectorXd ious = bbox_iou_xyah(z, tt_glmb_xyah);
        for (int i = 0; i < ious.size(); i++) {
            if (ious(i) > 0.2) { // consider as overlap with any existing tracks
                // compare re-id feature cdist with activating tracks
                double feat_dist = tt_glmb_feat.row(i) * feat;
                feat_dist /= (tt_glmb_feat.row(i).norm() * feat.norm());
                feat_dist = 1 - feat_dist;
                if (feat_dist < 0.2) {  // consider two re-identification features are similar
                    // new measurement and an existing track have similar feature, ignore this measurement
                    return true;
                }
            }
        }
        return false;
    }

    void clean_predict() {
        // hash label sets, find unique ones, merge all duplicates
        unordered_map<int, int> hypo_map;
        VectorXd glmb_temp_w = glmb_update_w;
        vector<VectorXi> glmb_temp_I(glmb_update_I);
        VectorXi glmb_temp_n = glmb_update_n;
        int unique_idx = 0;
        std::size_t seed;
        for (int hidx = 0; hidx < glmb_update_w.size(); hidx++) {
            VectorXi glmb_I = glmb_update_I[hidx];
            std::sort(glmb_I.data(), glmb_I.data() + glmb_I.size());
            seed = glmb_I.size();
            for (int &i: glmb_I) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            // If not present, then put it in unordered_set
            if (hypo_map.find(seed) == hypo_map.end()) {
                hypo_map[seed] = unique_idx;
                glmb_temp_w[unique_idx] = glmb_update_w[hidx];
                glmb_temp_I[unique_idx] = glmb_update_I[hidx];
                glmb_temp_n[unique_idx] = glmb_update_n[hidx];
                unique_idx += 1;
            } else {
                glmb_temp_w[hypo_map[seed]] += glmb_update_w[hidx];
            }
        }
        glmb_update_w = glmb_temp_w(seq(0, unique_idx - 1));  // 2
        glmb_temp_I.erase(glmb_temp_I.begin() + unique_idx, glmb_temp_I.end());
        glmb_update_I = glmb_temp_I;  // 3
        glmb_update_n = glmb_temp_n(seq(0, unique_idx - 1));  // 4
    }

    void clean_update(int time_step) {
        // flag used tracks
        VectorXi usedindicator = VectorXi::Zero(glmb_update_tt.size());
        for (int hidx = 0; hidx < glmb_update_w.size(); hidx++) {
            usedindicator(glmb_update_I[hidx]).array() += 1;
        }
        // remove unused tracks and reindex existing hypotheses/components
        VectorXi newindices = VectorXi::Zero(glmb_update_tt.size());
        int new_idx = 0;
        set<int> curr_tt_labels;
        vector<Target> glmb_clean_tt;
        for (int i = 0; i < newindices.size(); i++) {
            if (usedindicator(i) > 0) {
                newindices(i) = new_idx;
                new_idx += 1;
                glmb_clean_tt.push_back(glmb_update_tt[i]);
                // find unique current Targets (obtain all of its labels)
                if (curr_tt_labels.find(glmb_update_tt[i].l) == curr_tt_labels.end()) {
                    curr_tt_labels.insert(glmb_update_tt[i].l);
                }
            }
        }
        for (int hidx = 0; hidx < glmb_update_w.size(); hidx++) {
            glmb_update_I[hidx] = newindices(glmb_update_I[hidx]);
        }
        // remove pruned targets that are kept for 50 frames
        for (int i = prune_glmb_tt.size() - 1; i >= 0; i--) {
            if (time_step - prune_glmb_tt[i].last_activate > 50) {
                prune_glmb_tt.erase(prune_glmb_tt.begin() + i);
                prune_glmb_tt_feat.erase(prune_glmb_tt_feat.begin() + i);
                prune_glmb_tt_label.erase(prune_glmb_tt_label.begin() + i);
            }
        }
        // find pruned targets
        std::set<int> pruned_labels;
        std::set_difference(prev_tt_glmb_labels.begin(), prev_tt_glmb_labels.end(), curr_tt_labels.begin(),
                            curr_tt_labels.end(), std::inserter(pruned_labels, pruned_labels.begin()));
        for (Target tt: prev_glmb_update_tt) {
            if (pruned_labels.find(tt.l) != pruned_labels.end()) {
                prune_glmb_tt.push_back(tt);
                prune_glmb_tt_feat.push_back(tt.feat);
                prune_glmb_tt_label.push_back(tt.l);
            }
        }
        glmb_update_tt = glmb_clean_tt;
        prev_glmb_update_tt = glmb_clean_tt;
        prev_tt_glmb_labels = curr_tt_labels;
    }

    void prune() {
        // prune components with weights lower than specified threshold
        vector<int> idxkeep;
        vector<VectorXi> glmb_out_I;
        for (int i = 0; i < glmb_update_I.size(); i++) {
            if (glmb_update_w(i) > model.hyp_threshold) {
                idxkeep.push_back(i);
                glmb_out_I.push_back(glmb_update_I[i]);
            }
        }
        VectorXi idxkeep_eigen = VectorXi::Map(idxkeep.data(), idxkeep.size());
        VectorXd glmb_out_w = glmb_update_w(idxkeep_eigen);
        glmb_out_w /= glmb_out_w.sum();
        VectorXi glmb_out_n = glmb_update_n(idxkeep_eigen);
        VectorXd glmb_out_cdn(glmb_out_n.maxCoeff() + 1);
        for (int card = 0; card < glmb_out_cdn.size(); card++) {
            glmb_out_cdn[card] = (glmb_out_n.array() == card).select(glmb_out_w, 0).sum();
        }
        glmb_update_w = glmb_out_w;  // 2
        glmb_update_I = glmb_out_I;  // 3
        glmb_update_n = glmb_out_n;  // 4
        glmb_update_cdn = glmb_out_cdn;  // 5
    }

    void cap() {
        // cap total number of components to specified maximum
        if (glmb_update_w.size() > model.H_max) {
            // initialize original index locations
            vector<double> v_glmb_w(glmb_update_w.size());
            VectorXd::Map(&v_glmb_w[0], glmb_update_w.size()) = glmb_update_w;
            vector<int> idx(glmb_update_w.size());
            iota(idx.begin(), idx.end(), 0);
            stable_sort(idx.begin(), idx.end(),
                        [&v_glmb_w](int i1, int i2) { return v_glmb_w[i1] > v_glmb_w[i2]; });
            VectorXi idx_eigen = VectorXi::Map(idx.data(), idx.size());
            VectorXi idxkeep_eigen = idx_eigen(seq(0, model.H_max - 1));

            VectorXd glmb_out_w = glmb_update_w(idxkeep_eigen);
            VectorXi glmb_out_n = glmb_update_n(idxkeep_eigen);
            VectorXd glmb_out_cdn(glmb_out_n.maxCoeff() + 1);
            vector<VectorXi> glmb_out_I;
            for (int i: idxkeep_eigen) {
                glmb_out_I.push_back(glmb_update_I[i]);
            }

            for (int card = 0; card < glmb_out_cdn.size(); card++) {
                glmb_out_cdn[card] = (glmb_out_n.array() == card).select(glmb_out_n, 0).sum();
            }
            glmb_update_w = glmb_out_w;  // 2
            glmb_update_I = glmb_out_I;  // 3
            glmb_update_n = glmb_out_n;  // 4
            glmb_update_cdn = glmb_out_cdn;  // 5
        }
    }

public:
    GLMB(int width, int height, bool useFeat = true, bool useFuzzyPD = false) {
        glmb_update_w = VectorXd(1);
        glmb_update_w << 1;
        glmb_update_I.push_back(VectorXi(0));
        glmb_update_n = VectorXi(1);
        glmb_update_n(0) = 0;
        glmb_update_cdn = VectorXd(1);
        glmb_update_cdn(0) = 1;

        model = Model();
        region[0] = width;
        region[1] = height;
        pd_range[0] = 0.7;
        pd_range[1] = 0.95;
        murty = lap::Murty();
        id = 0;
        this->mUseFeat = useFeat;
        mUseFuzzyPD = useFuzzyPD; // either use Fuzzy PD (true) or IoA PD (false)
        if (useFuzzyPD) {
            mComputePD = ComputePD("./compute_pd.fis");
        }
        mAverageArea = 1;
    }

    std::tuple<MatrixXd, VectorXi> run_glmb(MatrixXd z, MatrixXd feat, int k) {
        // tlbr to cxcyah
        z(all, seq(2, 3)) -= z(all, seq(0, 1));
        z(all, seq(0, 1)) += z(all, seq(2, 3)) / 2;
        z.col(2) = z.col(2).array() / z.col(3).array();
        // joint prediction and update
        jointpredictupdate(model, z, feat, k);
        // pruning and truncation
        prune();
        cap();

        // extract estimates via recursive estimator, where trajectories are extracted via association history, and
        // track continuity is guaranteed with a non-trivial estimator

        // extract MAP cardinality and corresponding highest weighted component
        int M;
        glmb_update_cdn.maxCoeff(&M);
        int idxcmp;
        (glmb_update_w.array() * (glmb_update_n.array() == M).cast<double>()).maxCoeff(&idxcmp);
        MatrixXd X(4, M);
        VectorXi L(M);
        for (int m = 0; m < M; m++) {
            int idxptr = glmb_update_I[idxcmp](m);
            X.col(m) = glmb_update_tt[idxptr].m(seq(0, 3));
            L(m) = glmb_update_tt[idxptr].l;
        }
        if (M > 0 && mUseFuzzyPD) {
            VectorXd xrow2 = X.row(2);
            VectorXd xrow3 = X.row(3);
            mAverageArea = (xrow2.array() * xrow3.array().square()).sum() / M;
        }
        return {X, L};
    }

    std::tuple<MatrixXd, VectorXi, MatrixXd> run_glmb_feat(MatrixXd z, MatrixXd feat, int k) {
        // tlbr to cxcyah
        z(all, seq(2, 3)) -= z(all, seq(0, 1));
        z(all, seq(0, 1)) += z(all, seq(2, 3)) / 2;
        z.col(2) = z.col(2).array() / z.col(3).array();
        // joint prediction and update
        jointpredictupdate(model, z, feat, k);
        // pruning and truncation
        prune();
        cap();

        // extract estimates via recursive estimator, where trajectories are extracted via association history, and
        // track continuity is guaranteed with a non-trivial estimator

        // extract MAP cardinality and corresponding highest weighted component
        int M;
        glmb_update_cdn.maxCoeff(&M);
        int idxcmp;
        (glmb_update_w.array() * (glmb_update_n.array() == M).cast<double>()).maxCoeff(&idxcmp);
        MatrixXd X(4, M);
        VectorXi L(M);
        MatrixXd F(feat.cols(), M);
        for (int m = 0; m < M; m++) {
            int idxptr = glmb_update_I[idxcmp](m);
            X.col(m) = glmb_update_tt[idxptr].m(seq(0, 3));
            L(m) = glmb_update_tt[idxptr].l;
            F.col(m) = glmb_update_tt[idxptr].feat;
        }
        if (M > 0 && mUseFuzzyPD) {
            VectorXd xrow2 = X.row(2);
            VectorXd xrow3 = X.row(3);
            mAverageArea = (xrow2.array() * xrow3.array().square()).sum() / M;
        }
        return {X, L, F};
    }
};