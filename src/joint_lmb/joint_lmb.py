from scipy.spatial.distance import cdist
from scipy.stats.distributions import chi2
from scipy.special import logsumexp
import numpy as np
import lap

from utils import gate_meas_gms_idx, kalman_update_multiple, murty, kalman_predict_multiple, esf
from cpputils import bboxes_ioi_xyah_back2front, ComputePD, Murty, esf, bbox_iou_xyah, bboxes_ioi_xyah_back2front_all, \
    gibbs_jointpredupdt, MurtyMiller


class ModelParas:
    # filter parameters
    def __init__(self):
        self.T_max = 100  # maximum number of tracks
        self.track_threshold = 1e-3  # threshold to prune tracks
        self.H_upd = 500  # requested number of updated components/hypotheses (for GLMB update)

        self.x_dim = 8
        self.z_dim = 4
        self.P_G = 0.99  # gate size in percentage
        self.gamma = chi2.ppf(self.P_G, self.z_dim)  # inv chi^2 dn gamma value
        self.P_D = .8  # probability of detection in measurements
        self.P_S = .99  # survival/death parameters

        # clutter parameters
        self.lambda_c = 0.5  # poisson average rate of uniform clutter (per scan)
        self.range_c = np.array([[0, 1920], [0, 1080]])  # uniform clutter region
        self.pdf_c = 1 / np.prod(self.range_c[:, 1] - self.range_c[:, 0])  # uniform clutter density
        self.model_c = self.lambda_c * self.pdf_c

        float_precision = 'f8'
        # observation noise covariance
        self.R = np.array([[50., 0., 0., 0.],
                           [0., 50., 0., 0.],
                           [0., 0., 0.01, 0.],
                           [0., 0., 0., 50.]], dtype=float_precision)
        T = 1  # sate vector [x,y,a,h,dx,dy,da,dh]
        sigma_xy, sigma_a, sigma_h = 3 ** 2, 1e-4, 3 ** 2
        self.Q = np.array(
            [[T ** 4 * (sigma_xy / 4), 0, 0, 0, T ** 3 * (sigma_xy / 2), 0, 0, 0],  # process noise covariance
             [0, T ** 4 * (sigma_xy / 4), 0, 0, 0, T ** 3 * (sigma_xy / 2), 0, 0],
             [0, 0, T ** 4 * (sigma_a / 4), 0, 0, 0, T ** 3 * (sigma_a / 2), 0],
             [0, 0, 0, T ** 4 * (sigma_h / 4), 0, 0, 0, T ** 3 * (sigma_h / 2)],
             [T ** 3 * (sigma_xy / 2), 0, 0, 0, sigma_xy * T ** 2, 0, 0, 0],
             [0, T ** 3 * (sigma_xy / 2), 0, 0, 0, sigma_xy * T ** 2, 0, 0],
             [0, 0, T ** 3 * (sigma_a / 2), 0, 0, 0, sigma_a * T ** 2, 0],
             [0, 0, 0, T ** 3 * (sigma_h / 2), 0, 0, 0, sigma_h * T ** 2]], dtype=float_precision)

        self.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],  # Motion model: state transition matrix
                           [0, 1, 0, 0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]], dtype=float_precision)
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # observation matrix
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0]], dtype=float_precision)
        # Use P_birth from width, height of detected bbox diag(w, h, w, h)
        self.b_thresh = 0.95  # only birth a new target at a measurement that has lower assign_prob than this threshold
        self.lambda_b = 0.1  # Set lambda_b to the mean cardinality of the birth multi-Bernoulli RFS
        self.prob_birth = 0.03  # Initial existence probability of a birth track
    # END


class Target:
    # track table for GLMB (cell array of structs for individual tracks)
    # (1) r: existence probability
    # (2) Gaussian Mixture w (weight), m (mean), P (covariance matrix)
    # (3) Label: birth time & index of target at birth time step
    # (4) gatemeas: indexes gating measurement (using  Chi-squared distribution)
    # (5) deep feature: representing feature information extracted from re-identification network
    def __init__(self, z, feat, prob_birth, label, use_feat=True):
        x_dim = 8
        max_cpmt = 1000

        # wg, mg, Pg, ..., store temporary Gaussian mixtures while updating, see 'update_gms'
        self.wg = np.zeros(max_cpmt, dtype='f8')
        self.mg = np.zeros((x_dim, max_cpmt), dtype='f8')
        self.Pg = np.zeros((x_dim, x_dim, max_cpmt), dtype='f8')
        self.idxg = 0
        # store number of Gaussian mixtures before updating, see 'update_gms'
        self.gm_len = 0

        self.m = np.r_[z, np.zeros_like(z)][:, np.newaxis]
        wh2 = (z[3] + z[2] * z[3]) / 2  # half perimeter
        self.P = np.diag([wh2, wh2, 1, wh2, wh2, wh2, 1, wh2])[:, :, np.newaxis]
        self.alpha_feat = 0.9
        self.r = prob_birth  # existence probability of this birth
        self.w = np.ones(1)  # weights of Gaussians for birth track
        self.l = label  # label of this track
        self.r_max = 0  # maximum existence probability, use for hysteresis
        self.last_active = 0  # last frame, this track is not pruned or death
        self.use_feat = use_feat  # True/False whether using re-identification feature or NOT
        self.feat = None
        if use_feat:
            self.feat = feat
        self.gatemeas = np.empty(0, dtype=int)

    def predict_gms(self, model):
        self.r = model.P_S * self.r

        plength = self.m.shape[1]
        m_predict = np.zeros(self.m.shape)
        P_predict = np.zeros(self.P.shape)
        for idxp in range(plength):
            m_temp = np.dot(model.F, self.m[:, idxp])
            P_temp = model.Q + np.dot(model.F, np.dot(self.P[:, :, idxp], model.F.T))
            m_predict[:, idxp] = m_temp
            P_predict[:, :, idxp] = P_temp

        self.m = m_predict
        self.P = P_predict

    def update_gms(self, model, z, feat):
        #  =========== gating by tracks ===========
        zlength, plength = z.shape[0], self.m.shape[1]
        if zlength == 0:
            self.gatemeas = np.empty(0)
        else:
            valid_idx = np.zeros(zlength, dtype=bool)
            for j in range(plength):
                Sj = model.R + np.dot(np.dot(model.H, self.P[:, :, j]), model.H.T)
                Vs = np.linalg.cholesky(Sj)
                inv_sqrt_Sj = np.linalg.inv(Vs)
                nu = z.T - np.tile(np.dot(model.H, self.m[:, j].reshape(-1, 1)), zlength)
                dist = sum(np.square(np.dot(inv_sqrt_Sj, nu)))
                valid_idx = np.logical_or(valid_idx, dist < model.gamma)
            if self.use_feat:
                cdist_tmp = cdist(feat, self.feat[np.newaxis, :], metric='cosine').flatten()
                self.gatemeas = np.nonzero(np.logical_or(valid_idx, cdist_tmp < 0.3))[0]
                dist = cdist(self.feat[np.newaxis, :], feat[self.gatemeas])[0]
            else:
                self.gatemeas = np.nonzero(valid_idx)[0]
                dist = np.ones(len(self.gatemeas))
        #  ========================================

        self.idxg = 0
        # Gaussian mixtures for misdetection
        length = len(self.w)
        self.wg[:length] = self.w
        self.mg[:, :length] = self.m
        self.Pg[:, :, :length] = self.P
        self.idxg += length
        self.gm_len = length

        # Kalman update for each Gaussian with each gating measurement
        cost_update = np.zeros(len(self.gatemeas))
        for i, emm in enumerate(self.gatemeas):
            qz_temp, m_temp, P_temp = kalman_update_multiple(z[emm], model, self.m, self.P)
            w_temp = np.multiply(qz_temp, self.w) + np.spacing(1)

            pm_temp = 0.1 * dist[i] ** 14 + 0.9 * (2 - dist[i]) ** 14

            cost_update[i] = sum(w_temp) * pm_temp
            length = len(qz_temp)
            self.wg[self.idxg:self.idxg + length] = w_temp / sum(w_temp)
            self.mg[:, self.idxg:self.idxg + length] = m_temp
            self.Pg[:, :, self.idxg:self.idxg + length] = P_temp
            self.idxg += length

        # Copy values back to each fields
        self.w = self.wg[:self.idxg]
        self.m = self.mg[:, :self.idxg]
        self.P = self.Pg[:, :, :self.idxg]

        return cost_update

    # remove Gaussian mixtures in 'update_gm' that are not in ranked assignment
    def select_gms(self, select_idxs):
        self.w = self.wg[select_idxs]
        self.m = self.mg[:, select_idxs]
        self.P = self.Pg[:, :, select_idxs]

    def finalize_glmb2lmb(self, sums, association_idx, feat, time_step):
        if association_idx > 0:  # association_idx = 0, misdetection, keeping the same feature
            if self.use_feat:
                self.feat = self.alpha_feat * self.feat
                self.feat += (1 - self.alpha_feat) * feat[int(association_idx - 1), :]
                self.feat /= np.linalg.norm(self.feat)
            self.last_active = time_step  # only update if the highest hypothesis weight is not miss-detection
        repeat_sums = np.repeat(sums, self.gm_len)
        self.w *= repeat_sums
        self.r = sum(self.w)
        if self.r_max < self.r:
            self.r_max = self.r
        self.w = self.w / self.r

    def re_activate(self, z, feat, prob_birth):
        self.m = np.r_[z, np.zeros_like(z)][:, np.newaxis]
        wh2 = (z[3] + z[2] * z[3]) / 2
        self.P = np.diag([wh2, wh2, 1, wh2, wh2, wh2, 1, wh2])[:, :, np.newaxis]
        self.r = prob_birth
        self.w = np.ones(1)
        self.feat = feat

    def cleanup(self, elim_threshold=1e-5, l_max=10):
        # Gaussian prune, remove components that have weight lower than a threshold
        idx = np.nonzero(self.w > elim_threshold)[0]
        self.w = self.w[idx]
        self.m = self.m[:, idx]
        self.P = self.P[:, :, idx]
        # Gaussian cap, limit on number of Gaussians in each track
        if len(self.w) > l_max:
            idx = np.argsort(-self.w)
            w_new = self.w[idx[:l_max]]
            self.w = w_new * (sum(self.w) / sum(w_new))
            self.m = self.m[:, idx[:l_max]]
            self.P = self.P[:, :, idx[:l_max]]


class LMB:
    def __init__(self, use_feat=True):
        # initial prior
        self.tt_lmb = []
        self.tt_birth = []
        self.glmb_update_w = np.array([1])  # 2, vector of GLMB component/hypothesis weights
        self.assign_prob = None
        self.model = ModelParas()
        self.prune_tracks = []
        self.X = np.array([])
        self.L = np.array([], dtype=object)
        self.tt_lmb_xyah = np.array([])  # LMB tracks state [x,y,a,h]
        self.tt_lmb_feat = np.array([])  # LMB tracks reid feature
        self.pd = ComputePD('./compute_pd.fis')
        self.sampling = Murty()
        self.id = 0
        self.average_area = 1
        self.use_feat = use_feat

    def jointlmbpredictupdate(self, model, z, feat, k):
        #  generate birth tracks
        if k == 0:
            for idx in range(z.shape[0]):
                target = Target(z[idx], feat[idx], model.prob_birth, self.id, self.use_feat)
                self.id += 1  # or label = '0.' + str(idx)
                self.tt_birth.append(target)
            self.average_area = sum(z[:, 2] * z[:, 3] ** 2) / z.shape[0]

        # generate surviving tracks
        for target in self.tt_lmb:
            target.predict_gms(model)
        m = z.shape[0]  # number of measurements
        if m == 0:  # see MOT16-12, frame #445
            return  # no measurement to update, only predict existing tracks

        # create predicted tracks - concatenation of birth and survival
        self.tt_lmb += self.tt_birth  # copy track table back to GLMB struct
        ntracks = len(self.tt_lmb)
        self.tt_lmb_xyah = np.ascontiguousarray([tt.m[:4, np.argmax(tt.w)] for tt in self.tt_lmb], dtype=np.dtype('f8'))

        # compute Intersection Over Itself (between tt_lmb and estimated tracks) to find P_D for each track
        avps = np.zeros((ntracks, 1))
        avpd = np.zeros((ntracks, 1))
        # object stands close to a camera has a higher bottom coordinate [(0, 0) : (top, left)]
        # back to front: objects from far to near a camera
        tt_labels = np.array([tt.l for tt in self.tt_lmb], dtype=np.dtype('int'))
        mutual_ioi = bboxes_ioi_xyah_back2front(self.tt_lmb_xyah, tt_labels, self.X.T, self.L)
        self.tt_lmb_xyah[:, 2] = np.clip(self.tt_lmb_xyah[:, 2], 0.15, None)  # constraint 'a' not to be negative
        area_all = self.tt_lmb_xyah[:, 2] * self.tt_lmb_xyah[:, 3] ** 2
        area_rate = np.clip(area_all / self.average_area, 0, 2)
        for tabidx, tabidx_ioa in enumerate(mutual_ioi):
            # average detection/missed probabilities
            avpd[tabidx] = self.pd.compute(area_rate[tabidx], tabidx_ioa)
            # average survival/death probabilities
            avps[tabidx] = self.tt_lmb[tabidx].r
        avqd = 1 - avpd
        avqs = 1 - avps

        # create updated tracks (single target Bayes update)
        allcostm = np.zeros((ntracks, m))
        for tabidx in range(ntracks):
            cost_update = self.tt_lmb[tabidx].update_gms(model, z, feat)
            allcostm[tabidx, self.tt_lmb[tabidx].gatemeas] = cost_update

        # joint cost matrix, eta_j eq (22) "An Efficient Implementation of the GLMB"
        eta_j = np.multiply(np.multiply(avps, avpd), allcostm) / model.model_c
        jointcostm = np.zeros((ntracks, 2 * ntracks + m))
        np.fill_diagonal(jointcostm, avqs)
        np.fill_diagonal(jointcostm[:, ntracks:], np.multiply(avps, avqd))
        jointcostm[:, 2 * ntracks:2 * ntracks + m] = eta_j

        # calculate best updated hypotheses/components
        # murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
        if jointcostm.shape[0] > 0:
            uasses, nlcost = self.sampling.draw_solutions(-np.log(jointcostm), model.H_upd)
        else:
            # no need sampling for empty cost matrix
            uasses, nlcost = np.empty(0), np.empty(0)
        uasses = uasses + 1
        uasses[uasses <= ntracks] = -np.inf  # set not born/track deaths to -inf assignment
        uasses[(uasses > ntracks) & (uasses <= 2 * ntracks)] = 0  # set survived+missed to 0 assignment
        # set survived+detected to assignment of measurement index from 1:|Z|
        uasses[uasses > 2 * ntracks] = uasses[uasses > 2 * ntracks] - 2 * ntracks

        # component updates
        glmb_nextupdate_w = np.zeros(len(nlcost))
        self.assign_prob = np.zeros(m)  # adaptive birth weight for each measurement
        assign_meas = np.zeros((m, len(nlcost)), dtype=int)  # store indexes of measurement assigned to a track

        iois = bboxes_ioi_xyah_back2front_all(self.tt_lmb_xyah)
        self.pd.set_recompute_cost(avqs, avps, allcostm, model.model_c)
        # generate corrresponding jointly predicted/updated hypotheses/components
        for hidx in range(0, len(nlcost)):
            update_hypcmp_tmp = uasses[hidx, :]
            new_cost = self.pd.recompute_cost(update_hypcmp_tmp, iois, area_all)
            # hypothesis/component weight
            # Vo Ba-Ngu "An efficient implementation of the generalized labeled multi-Bernoulli filter." eq (20)
            omega_z = -model.lambda_c + m * np.log(model.model_c) - new_cost  # nlcost[hidx]
            # Get measurement index from uasses (make sure minus 1 from [mindices+1])
            meas_idx = update_hypcmp_tmp[update_hypcmp_tmp > 0].astype(int) - 1
            assign_meas[meas_idx, hidx] = 1
            glmb_nextupdate_w[hidx] = omega_z

        glmb_nextupdate_w = np.exp(glmb_nextupdate_w - logsumexp(glmb_nextupdate_w))  # normalize weights

        self.assign_prob = assign_meas @ glmb_nextupdate_w

        # The following implementation is optimized for GLMB to LMB (glmb2lmb)
        # Refer "The Labeled Multi-Bernoulli Filter, 2014"
        for (i, target) in enumerate(self.tt_lmb):
            notinf_uasses_idxs = np.nonzero(uasses[:, i] >= 0)[0]
            workon_uasses = uasses[notinf_uasses_idxs, i]
            workon_weights = glmb_nextupdate_w[notinf_uasses_idxs]

            u, inv = np.unique(workon_uasses, return_inverse=True)
            if len(u) == 0:  # no measurement association (including misdetection)
                continue
            sums = np.zeros(len(u), dtype=workon_weights.dtype)
            np.add.at(sums, inv, workon_weights)

            # select gating measurement indexes appear in ranked assignment (u: variable)
            # 0 for mis detection, 1->n for measurement index
            _, select_idxs, _ = np.intersect1d(np.insert(target.gatemeas + 1, 0, 0), u, return_indices=True)

            # select 'block of gaussian mixtures' that are in 'select_idxs'
            l_range = np.tile(np.arange(target.gm_len), len(select_idxs))
            start_idx = np.repeat(select_idxs, target.gm_len) * target.gm_len
            select_idxs = l_range + start_idx
            target.select_gms(select_idxs)

            target.finalize_glmb2lmb(sums, u[np.argmax(sums)], feat, time_step=k)

        # create birth tracks
        self.apdative_birth(self.assign_prob, z, feat, model, k)

    def re_activate_tracks(self, z, feat, model, prob_birth):
        if not self.use_feat:
            return False
        if len(self.prune_tracks) > 0:
            track_features = np.asarray([target.feat for target in self.prune_tracks])
            feats_dist = cdist(track_features, feat[np.newaxis, :], metric='cosine')
            if np.amin(feats_dist) < 0.25:  # pruned track cannot update feature for few frames
                idx = np.argmin(feats_dist)
                tt = self.prune_tracks[idx]
                gating = gate_meas_gms_idx(z[np.newaxis, :], feat[np.newaxis, :], model, tt.m, tt.P, tt.feat)
                if len(gating) > 0:  # associated measurement must be closed to a pruned track
                    tt.re_activate(z, feat, prob_birth)
                    self.tt_lmb.append(tt)
                    self.prune_tracks.remove(tt)
                    return True
        false_meas = False
        # first, checking whether new measurement overlap with existing tracks
        ious = bbox_iou_xyah(z, self.tt_lmb_xyah)
        iou_idx = np.nonzero(ious > 0.2)[0]
        if len(iou_idx):  # consider as overlap with any existing tracks
            # second, compare re-id feature cdist with activating tracks
            track_features = np.asarray([self.tt_lmb[idx].feat for idx in iou_idx])
            feats_dist = cdist(track_features, feat[np.newaxis, :], metric='cosine')
            if np.amin(feats_dist) < 0.2:  # consider two re-identification features are similar
                # new measurement and an existing track have similar feature, ignore this measurement
                false_meas = True
        return false_meas

    def reappear_tracks(self, assign_prob, z, feat, b_idx, model):
        re_activate = False
        if len(self.prune_tracks) == 0 or len(feat) == 0:
            return re_activate
        track_features = np.asarray([target.feat for target in self.prune_tracks])
        tt_feat_dist = cdist(track_features, feat)
        tt_feat_dist = 0.05 * tt_feat_dist * 2 + 0.95 * (2 - tt_feat_dist) * 2
        cost = tt_feat_dist * np.tile(1 - assign_prob, (len(self.prune_tracks), 1))
        assignment_index = lap.lapjv(-np.log(cost), extend_cost=True, cost_limit=-0.5)
        b_idx_select = np.ones(len(b_idx), dtype=bool)
        for tt_idx, meas_idx in enumerate(assignment_index[1]):
            if meas_idx < 0:
                continue
            target = self.prune_tracks[tt_idx]
            target.re_activate(z[meas_idx], feat[meas_idx],
                               min(model.prob_birth, cost[tt_idx, meas_idx] / np.sum(cost)))
            self.tt_birth.append(target)
            b_idx_select[meas_idx] = False
        return b_idx_select

    def apdative_birth(self, assign_prob, z, feat, model, k):
        not_assigned_sum = sum(1 - assign_prob) + np.spacing(1)  # make sure this sum is not zero
        b_idx = np.nonzero(assign_prob < self.model.b_thresh)[0]
        self.tt_birth = []
        for idx, meas_idx in enumerate(b_idx):
            # eq (75) "The Labeled Multi-Bernoulli Filter", Stephan Reuter∗, Ba-Tuong Vo, Ba-Ngu Vo, ...
            prob_birth = min(model.prob_birth, (1 - assign_prob[meas_idx]) / not_assigned_sum * model.lambda_b)
            prob_birth = max(prob_birth, np.spacing(1))  # avoid zero birth probability

            re_activate = self.re_activate_tracks(z[meas_idx], feat[meas_idx], model, prob_birth)
            if re_activate:
                continue

            target = Target(z[meas_idx], feat[meas_idx], prob_birth, self.id, self.use_feat)
            self.id += 1  # or label = str(k + 1) + '.' + str(idx)
            self.tt_birth.append(target)
        # END

    def clean_lmb(self, model, tim_step):
        # prune tracks with low existence probabilities

        # extract vector of existence probabilities from LMB track table
        rvect = np.array([tt.r for tt in self.tt_lmb])

        idxkeep = np.nonzero(rvect > model.track_threshold)[0]
        tt_lmb_out = [self.tt_lmb[i] for i in idxkeep]
        idxprune = np.nonzero(rvect <= model.track_threshold)[0]
        self.prune_tracks = self.prune_tracks + [self.tt_lmb[i] for i in idxprune]
        remove = []
        for t in self.prune_tracks:
            if tim_step - t.last_active > 50:
                remove.append(t)
        for t in remove:
            self.prune_tracks.remove(t)

        # cleanup tracks
        for target in tt_lmb_out:
            target.cleanup()

        self.tt_lmb = tt_lmb_out

    # END clean_lmb

    def extract_estimates(self):
        # extract estimates via MAP cardinality and corresponding tracks
        num_tracks = len(self.tt_lmb)
        rvect = np.array([tt.r for tt in self.tt_lmb])
        rvect = np.minimum(rvect, 1. - 1e-6)
        rvect = np.maximum(rvect, 1e-6)
        # Calculate the cardinality distribution of the multi-Bernoulli RFS
        cdn = esf(rvect / (1 - rvect))  # np.prod(1 - rvect) * esf(rvect / (1 - rvect))
        mode = np.argmax(cdn)
        N = min(len(rvect), mode)
        idxcmp = np.argsort(-rvect)
        X, L = np.zeros((4, num_tracks)), np.zeros(num_tracks, dtype=int)
        select_idx = 0
        for n in range(N):
            select_target = self.tt_lmb[idxcmp[n]]
            idxtrk = np.argmax(select_target.w)
            X[:, select_idx] = select_target.m[:4, idxtrk]
            L[select_idx] = select_target.l
            select_idx += 1

        # hysteresis, eq (71) "The Labeled Multi-Bernoulli Filter", Stephan Reuter∗, Ba-Tuong Vo, Ba-Ngu Vo, ...
        for idxx in range(N, num_tracks):
            select_target = self.tt_lmb[idxcmp[idxx]]
            if select_target.r_max > 0.7 and select_target.r > 0.1:
                idxtrk = np.argmax(select_target.w)
                X[:, select_idx] = select_target.m[:4, idxtrk]
                L[select_idx] = select_target.l
                select_idx += 1
        X[2, :] = np.clip(X[2, :], 0.15, None)  # constraint 'a' not to be negative
        if select_idx > 0:
            self.average_area = sum(X[2, :] * X[3, :] ** 2) / select_idx
        return X[:, :select_idx], L[:select_idx]

    def run_lmb(self, z, feat, k):
        # Input z : (n, 4), number of measurements and Top, Left, Bottom, Right of a bounding box
        # Input feat : (n, reid_dim), z[i, :] has re-id feature feat[i, :], norm2 feature
        # tlbr to cxcyah
        z[:, 2:4] -= z[:, 0:2]
        z[:, 0:2] += z[:, 2:4] / 2
        z[:, 2] = z[:, 2] / z[:, 3]

        # joint predict and update, results in GLMB, convert to LMB
        self.jointlmbpredictupdate(self.model, z, feat, k)

        # pruning, truncation and track cleanup
        self.clean_lmb(self.model, k)

        # state estimation
        X, L = self.extract_estimates()
        self.X = X
        self.L = L
        return np.copy(X), L
    # END
