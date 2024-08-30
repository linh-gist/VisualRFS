from scipy.spatial.distance import cdist
from scipy.stats.distributions import chi2
from scipy.stats import multivariate_normal
import numpy as np
from copy import deepcopy
from scipy.special import logsumexp
from cpputils import ComputePD, Murty, esf, bbox_iou_xyah, bboxes_ioi_xyah_back2front_all, gibbs_jointpredupdt


class ModelParas:
    # filter parameters
    def __init__(self):
        self.H_upd = 500  # requested number of updated components/hypotheses
        self.H_max = 500  # cap on number of posterior components/hypotheses
        self.hyp_threshold = 1e-10  # pruning threshold for components/hypotheses

        self.z_dim = 4
        self.P_G = 0.95  # gate size in percentage
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
        self.R = np.array([[70., 0., 0., 0.],
                           [0., 30., 0., 0.],
                           [0., 0., 0.0055, 0.],
                           [0., 0., 0., 70.]], dtype=float_precision)
        T = 1  # sate vector [x,y,a,h,dx,dy,da,dh]
        sigma_xy, sigma_a, sigma_h = 5 ** 2, 1e-4, 5 ** 2
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
        self.w_birth = 1
        self.b_thresh = 0.95  # only birth a new target at a measurement that has lower assign_prob than this threshold
        self.lambda_b = 0.1  # Set lambda_b to the mean cardinality of the birth multi-Bernoulli RFS
        self.prob_birth = 5e-4  # Initial existence probability of a birth track
        # NOTE: prob_birth=0.03, lambda_b=0.1 need to be chosen carefully so that tt.r is not too small (e.g. 5e-4)
        # NOTE: init covariance, P=diag(50) and lamda_c=0.5 are not too high (e.g. P=diag(1000), lamda_c=8)
        # NOTE: P: high => uncertainty high => qz low, tt.r small => cannot create birth tracks
    # END


class Target:
    # track table for GLMB (cell array of structs for individual tracks)
    # (1) P_S: survival probability
    # (2) P_D: detection probability
    # (3) r:   birth probability, adaptive birth
    # (4) Gaussian component m (mean), P (covariance matrix), Not implemented mixture with w (weight)
    # (5) Label: birth time & index of target at birth time step
    # (6) gatemeas: indexes gating measurement (using  Chi-squared distribution), reid feature gating
    # (7) feat: re-identification feature (obtained from Deep Learning framework)

    def __init__(self, z, feat, prob_birth, birth_time, label, region, use_feat=True):
        self.m = np.r_[z, np.zeros_like(z)]
        wh2 = 50  # (z[3] + z[2] * z[3]) / 2  # half perimeter
        self.P = np.diag([wh2, wh2, 1, wh2, wh2, wh2, 1, wh2])  # covs of Gaussians for birth track
        self.alpha_feat = 0.9
        self.r = prob_birth
        self.l = label  # or str(birth_time) + '.' + str(target_idx)  # track label
        self.last_active = 0  # last frame, this track is not pruned or death
        self.P_S = 0.99
        self.region = region
        self.birth_time = birth_time
        self.use_feat = use_feat  # True/False whether using re-identification feature or NOT
        self.feat = None
        if use_feat:
            self.feat = feat
        self.gatemeas = np.empty(0, dtype=int)

    def predict(self, model, time_step, average_area):
        m_predict = np.dot(model.F, self.m)  # kalman predict single
        self.P = model.Q + np.dot(model.F, np.dot(self.P, model.F.T))
        self.m = m_predict
        w = max(0.2, m_predict[2]) * m_predict[3]
        t, l = m_predict[1] - m_predict[3] / 2, m_predict[0] - w / 2
        b, r = m_predict[1] + m_predict[3] / 2, m_predict[0] + w / 2
        if r > 0 and b > 0 and t < self.region[1] and l < self.region[0]:
            # unknown scene mask, b(x) = 0.99 for target inside image
            # A labeled random finite set online multi-object tracker for video data, Du Yong Kim, eq(11)
            P_S = 0.95 / (1 + np.exp(0.75 * (self.birth_time - time_step)))
            self.P_S = P_S / (1 + np.exp(3.5 * (0.1 - w * m_predict[3] / average_area)))
        else:
            # lower survival probability for target outside of image [0, 0, width, height]
            self.P_S = 0.1

    def update(self, model, z, feat, time_step):
        self.last_active = time_step
        tt_update = deepcopy(self)
        #  =========== kalman update single ===========
        mu = np.dot(model.H, self.m)
        S = model.R + np.dot(np.dot(model.H, self.P), model.H.T)
        Vs = np.linalg.cholesky(S)
        inv_sqrt_S = np.linalg.inv(Vs)
        iS = np.dot(inv_sqrt_S, inv_sqrt_S.T)
        K = np.dot(np.dot(self.P, model.H.T), iS)
        z_mu = z - mu
        qz_temp = multivariate_normal.pdf(z, mean=mu, cov=S)  # g(z|x)
        m_temp = self.m + np.dot(K, z_mu)
        P_temp = np.dot((np.eye(len(self.P)) - np.dot(K, model.H)), self.P)
        #  ============================================
        tt_update.m = m_temp
        tt_update.P = P_temp

        pm_temp = 1
        if self.use_feat:
            # npdist = np.linalg.norm(self.feat - feat)  # euclidean of two norm vectors
            npdist = cdist(self.feat[np.newaxis, :], feat[np.newaxis, :])[0, 0]
            feat_temp = self.alpha_feat * self.feat + (1 - self.alpha_feat) * feat
            tt_update.feat = feat_temp / np.linalg.norm(feat_temp)
            # consider pm in prediction is [0.01, 0.99]
            pm_temp = 0.01 * npdist ** 15 + 0.99 * (2 - npdist) ** 15
        cost_update = qz_temp * pm_temp + np.spacing(1)

        return cost_update, tt_update

    def gating(self, model, z, feat):
        zlength = z.shape[0]
        if zlength == 0:
            return np.empty(0)
        Sj = model.R + np.dot(np.dot(model.H, self.P), model.H.T)
        Vs = np.linalg.cholesky(Sj)
        inv_sqrt_Sj = np.linalg.inv(Vs)
        nu = z - np.tile(np.dot(model.H, self.m), (zlength, 1))
        dist = sum(np.square(np.dot(inv_sqrt_Sj, nu.T)))
        if self.use_feat:
            # similar reid features have cosine distance less than 0.3
            sscore = feat.dot(self.feat)
            sscore = 1 - sscore / (np.linalg.norm(feat, axis=1) * np.linalg.norm(self.feat))
            self.gatemeas = np.nonzero(np.logical_or(dist < model.gamma, sscore < 0.3))[0]
        else:
            self.gatemeas = np.nonzero(dist < model.gamma)[0]

    def re_activate(self, z, feat, prob_birth, time_step):
        self.m = np.r_[z, np.zeros_like(z)]
        wh2 = (z[3] + z[2] * z[3]) / 2
        self.P = np.diag([wh2, wh2, 1, wh2, wh2, wh2, 1, wh2])
        self.r = prob_birth
        self.feat = feat
        self.last_active = time_step
    #  END


class GLMB:  # delta GLMB
    def __init__(self, width, height, use_feat=True):
        # initial Numpy Data type for target
        self.glmb_update_tt = []  # (1) track table for GLMB (individual tracks)
        self.glmb_update_w = np.array([1])  # (2) vector of GLMB component/hypothesis weights
        self.glmb_update_I = [np.array([], dtype=int)]  # (3) cell of GLMB component/hypothesis labels (in track table)
        self.glmb_update_n = np.array([0])  # (4) vector of GLMB component/hypothesis cardinalities
        self.glmb_update_cdn = np.array([1])  # (5) cardinality distribution of GLMB

        self.assign_prob = None
        self.model = ModelParas()
        np.set_printoptions(linewidth=2048)  # use in clean_predict() to convert array2string

        self.tt_labels = np.empty([])
        self.tt_prune = []
        self.tt_glmb_xyah = np.array([])  # GLMB tracks state [x,y,a,h]
        self.tt_glmb_feat = np.array([])  # GLMB tracks reid feature

        self.prev_tt_glmb_labels = np.array([])
        self.prev_glmb_update_tt = []
        self.prune_glmb_tt = []
        self.prune_glmb_tt_feat = []
        self.prune_glmb_tt_label = []
        self.region = [width, height]
        self.pd = ComputePD("./compute_pd.fis")
        self.sampling = Murty()
        self.id = 0
        self.average_area = 1
        self.use_feat = use_feat

    def jointpredictupdate(self, model, z, feat, k):
        # create birth tracks
        if k == 0:
            self.tt_birth = []
            for idx in range(z.shape[0]):
                tt = Target(z[idx], feat[idx], model.prob_birth, k, self.id, self.region, self.use_feat)
                self.id += 1
                self.tt_birth.append(tt)
            self.average_area = sum(z[:, 2] * z[:, 3] ** 2) / z.shape[0]

        # create surviving tracks - via time prediction (single target CK)
        for tt in self.glmb_update_tt:
            tt.predict(model, k, self.average_area)
        m = z.shape[0]  # number of measurements
        if m == 0:  # see MOT16-12, frame #445
            return  # no measurement to update, only predict existing tracks

        # create predicted tracks - concatenation of birth and survival
        glmb_predict_tt = self.tt_birth + self.glmb_update_tt  # copy track table back to GLMB struct
        self.tt_glmb_xyah = np.ascontiguousarray([tt.m for tt in glmb_predict_tt], dtype=np.dtype('f8'))
        self.tt_glmb_feat = np.ascontiguousarray([tt.feat for tt in glmb_predict_tt], dtype=np.dtype('f8'))
        cpreds = len(glmb_predict_tt)

        # precalculation loop for average survival/death probabilities
        avps = np.array([tt.r for tt in self.tt_birth] +
                        [tt.P_S for tt in self.glmb_update_tt])[:, np.newaxis]
        avqs = 1 - avps

        # create updated tracks (single target Bayes update)
        # missed detection tracks (legacy tracks) using deepcopy
        tt_update = deepcopy(glmb_predict_tt) + [[] for i in range((m) * cpreds)]  # initialize cell array

        # measurement updated tracks (all pairs)
        allcostm = np.zeros((cpreds, m))
        for tabidx, tt in enumerate(glmb_predict_tt):
            glmb_predict_tt[tabidx].gating(model, z, feat)
            for i, emm in enumerate(glmb_predict_tt[tabidx].gatemeas):
                stoidx = cpreds * (emm + 1) + tabidx
                cost_update, tt_update_gate = tt.update(model, z[emm], feat[emm], k)
                allcostm[tabidx, emm] = cost_update
                tt_update[stoidx] = tt_update_gate

        # joint cost matrix, eta_j is the cost matrix for survived and detected tracks
        eta_j = np.multiply(avps, allcostm) / (model.lambda_c * model.pdf_c)
        jointcostm = np.zeros((cpreds, 2 * cpreds + m))
        np.fill_diagonal(jointcostm, avqs)
        np.fill_diagonal(jointcostm[:, cpreds:], avps)
        jointcostm[:, 2 * cpreds:] = eta_j

        # gated measurement index matrix
        gatemeasidxs = -1 * np.ones((cpreds, m), dtype=int)
        for tabidx, tt in enumerate(glmb_predict_tt):
            gatemeasidxs[tabidx, tt.gatemeas] = tt.gatemeas
        # component updates
        runidx = 0
        glmb_nextupdate_w = np.zeros(model.H_upd * 2)
        glmb_nextupdate_I = []
        glmb_nextupdate_n = np.zeros(model.H_upd * 2, dtype=int)
        assign_meas = np.zeros((m, model.H_upd * 2), dtype=int)  # use to normalize assign_prob using glmb_nextupdate_w
        nbirths = len(self.tt_birth)
        hypoth_num = np.rint(model.H_upd * np.sqrt(self.glmb_update_w) / sum(np.sqrt(self.glmb_update_w))).astype(int)
        neglog_jointcostm = -np.log(jointcostm)  # negative log cost

        for pidx in range(0, len(self.glmb_update_w)):
            # calculate best updated hypotheses/components
            nexists = len(self.glmb_update_I[pidx])
            ntracks = nbirths + nexists
            # indices of all births and existing tracks  for current component
            tindices = np.concatenate((np.arange(0, nbirths), nbirths + self.glmb_update_I[pidx]))
            if len(tindices) == 0:
                continue
            # union indices of gated measurements for corresponding tracks
            mindices = np.unique(gatemeasidxs[tindices, :])
            if -1 in mindices:
                mindices = mindices[1:]
            # cost matrix - [no_birth/is_death | born/survived+missed | born/survived+detected]
            take_rows = neglog_jointcostm[tindices]
            neglogcostm = np.copy(take_rows[:, np.concatenate((tindices, cpreds + tindices, 2 * cpreds + mindices))])

            # calculate average detection / missed probabilities
            glmbI_xyah = np.ascontiguousarray([glmb_predict_tt[tt_i].m for tt_i in tindices],
                                              dtype=np.dtype('f8'))
            avpd = np.zeros((ntracks, 1))
            # object stands close to a camera has a higher bottom coordinate [(0, 0) : (top, left)]
            # back to front: objects from far to near a camera
            mutual_ioi = bboxes_ioi_xyah_back2front_all(glmbI_xyah)
            overlapped_ioi = np.amax(mutual_ioi, axis=1)
            glmbI_xyah[:, 2] = np.clip(glmbI_xyah[:, 2], 0.2, None)  # constraint 'a' not to be negative
            area_all = glmbI_xyah[:, 2] * glmbI_xyah[:, 3] ** 2
            area_rate = np.clip(area_all / self.average_area, 0, 2)
            for tabidx, tabidx_ioa in enumerate(overlapped_ioi):
                # average detection/missed probabilities
                avpd[tabidx] = self.pd.compute(area_rate[tabidx], tabidx_ioa)
            neglogcostm[:, ntracks:2 * ntracks] -= np.log(1 - avpd)
            neglogcostm[:, 2 * ntracks:] -= np.log(avpd)

            # murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
            # output theta, measurement to track association
            uasses, nlcost = self.sampling.draw_solutions(neglogcostm, hypoth_num[pidx])
            # uasses, nlcost = gibbs_jointpredupdt(neglogcostm, hypoth_num[pidx])  # use Gibbs Sampling
            uasses = uasses + 1
            uasses[uasses <= ntracks] = -np.inf  # set not born/track deaths to -inf assignment
            uasses[(uasses > ntracks) & (uasses <= 2 * ntracks)] = 0  # set survived+missed to 0 assignment
            # set survived+detected to assignment of measurement index from 1:|Z|
            uasses[uasses > 2 * ntracks] = uasses[uasses > 2 * ntracks] - 2 * ntracks
            # restore original indices of gated measurements
            uasses[uasses > 0] = mindices[uasses[uasses > 0].astype(int) - 1] + 1

            # generate corrresponding jointly predicted/updated hypotheses/components
            len_nlcost = len(nlcost)
            # hypothesis/component cardinality
            if len_nlcost:
                glmb_nextupdate_n[runidx:runidx + len_nlcost] = np.sum(uasses >= 0, axis=1)
            self.pd.set_recompute_cost(avqs[tindices], avps[tindices], allcostm[tindices, :], model.model_c)
            start_runidx = runidx
            for hidx in range(len_nlcost):
                update_hypcmp_tmp = uasses[hidx, :]
                update_hypcmp_idx = cpreds * update_hypcmp_tmp + np.concatenate(
                    (np.arange(0, nbirths), nbirths + self.glmb_update_I[pidx])).astype(int)
                # Get measurement index from uasses (make sure minus 1 from [mindices+1])
                uasses_idx = update_hypcmp_tmp[update_hypcmp_tmp > 0].astype(int) - 1
                assign_meas[uasses_idx, runidx] = 1  # Setting index of measurements associate with a track
                # hypothesis/component tracks (via indices to track table)
                glmb_nextupdate_I.append(update_hypcmp_idx[update_hypcmp_idx >= 0].astype(int))

                nlcost[hidx] = self.pd.recompute_cost(update_hypcmp_tmp, mutual_ioi, area_all)
                runidx = runidx + 1
            # hypothesis/component weight eqs (20) => (15) => (17) => (4), omega_z
            # Vo Ba-Ngu "An efficient implementation of the generalized labeled multi-Bernoulli filter."
            glmb_nextupdate_w[start_runidx:start_runidx + len_nlcost] = -model.lambda_c + m * np.log(
                model.model_c) + np.log(self.glmb_update_w[pidx]) - nlcost

        glmb_nextupdate_w = glmb_nextupdate_w[:runidx]
        glmb_nextupdate_n = glmb_nextupdate_n[:runidx]
        glmb_nextupdate_w = np.exp(glmb_nextupdate_w - logsumexp(glmb_nextupdate_w))  # normalize weights
        assign_prob = assign_meas[:, :runidx] @ glmb_nextupdate_w  # adaptive birth weight for each measurement
        # create birth tracks
        self.apdative_birth(z, feat, assign_prob, model, k)

        # extract cardinality distribution
        glmb_nextupdate_cdn = np.zeros(max(glmb_nextupdate_n) + 1)
        for card in range(0, max(glmb_nextupdate_n) + 1):
            glmb_nextupdate_cdn[card] = sum(
                glmb_nextupdate_w[glmb_nextupdate_n == card])  # extract probability of n targets

        # copy glmb update to the next time step
        self.glmb_update_tt = tt_update  # 1
        self.glmb_update_w = glmb_nextupdate_w  # 2
        self.glmb_update_I = glmb_nextupdate_I  # 3
        self.glmb_update_n = glmb_nextupdate_n  # 4
        self.glmb_update_cdn = glmb_nextupdate_cdn  # 5

        # remove duplicate entries and clean track table
        self.clean_predict()
        self.clean_update(k)

    def apdative_birth(self, z, feat, assign_prob, model, k):
        not_assigned_sum = sum(1 - assign_prob) + np.spacing(1)  # make sure this sum is not zero
        b_idx = np.nonzero(assign_prob <= self.model.b_thresh)[0]
        self.tt_birth = []
        for idx, meas_idx in enumerate(b_idx):
            # eq (75) "The Labeled Multi-Bernoulli Filter", Stephan Reuter∗, Ba-Tuong Vo, Ba-Ngu Vo, ...
            prob_birth = min(model.prob_birth, (1 - assign_prob[meas_idx]) / not_assigned_sum * model.lambda_b)
            prob_birth = max(prob_birth, np.spacing(1))  # avoid zero birth probability

            re_activate = self.re_activate_tracks(z[meas_idx][:4], feat[meas_idx], prob_birth, k)
            if re_activate:
                continue

            tt = Target(z[meas_idx], feat[meas_idx], prob_birth, k + 1, self.id, self.region, self.use_feat)
            self.id += 1
            self.tt_birth.append(tt)
        # END

    def re_activate_tracks(self, z, feat, prob_birth, k):
        if not self.use_feat:
            return False
        if len(self.prune_glmb_tt) > 0:
            track_features = np.asarray(self.prune_glmb_tt_feat)
            # feats_dist = track_features.dot(feat)  # cosine distance between vectors
            # feats_dist = 1 - feats_dist / (np.linalg.norm(track_features, axis=1) * np.linalg.norm(feat))
            feats_dist = cdist(track_features, feat[np.newaxis, :], metric='cosine')
            if np.amin(feats_dist) < 0.25:  # pruned track cannot update feature for few frames
                idx = np.argmin(feats_dist)
                target = self.prune_glmb_tt[idx]
                same_label_idxs = np.nonzero(np.asarray(self.prune_glmb_tt_label) == target.l)[0]
                target.re_activate(z, feat, prob_birth, k)
                self.tt_birth.append(target)
                same_label_idxs = sorted(same_label_idxs, reverse=True)
                for same_idx in same_label_idxs:
                    del self.prune_glmb_tt[same_idx]
                    del self.prune_glmb_tt_feat[same_idx]
                    del self.prune_glmb_tt_label[same_idx]
                return True  # recall this Target
        # Clean false measurements
        false_meas = False
        # first, checking whether new measurement overlap with existing tracks
        ious = bbox_iou_xyah(z, self.tt_glmb_xyah)
        iou_idx = np.nonzero(ious > 0.2)[0]
        if len(iou_idx):  # consider as overlap with any existing tracks
            # second, compare re-id feature cdist with activating tracks
            track_features = self.tt_glmb_feat[iou_idx]
            # feats_dist = track_features.dot(feat)
            # feats_dist = 1 - feats_dist / (np.linalg.norm(track_features, axis=1) * np.linalg.norm(feat))
            feats_dist = cdist(track_features, feat[np.newaxis, :], metric='cosine')
            if np.amin(feats_dist) < 0.2:  # consider two re-identification features are similar
                false_meas = True  # new measurement and an existing track have similar feature, ignore this measurement
        return false_meas

    def clean_predict(self):
        # hash label sets, find unique ones, merge all duplicates
        glmb_raw_hash = np.empty(len(self.glmb_update_w), dtype=np.dtype('<U2048'))
        for hidx in range(0, len(self.glmb_update_w)):
            hash_str = np.array2string(np.sort(self.glmb_update_I[hidx]), separator='*')[1:-1]
            glmb_raw_hash[hidx] = hash_str

        cu, _, ic = np.unique(glmb_raw_hash, return_index=True, return_inverse=True, axis=0)

        glmb_temp_w = np.zeros((len(cu)))
        glmb_temp_I = [np.array([]) for i in range(0, len(ic))]
        glmb_temp_n = np.zeros((len(cu)), dtype=int)
        for hidx in range(0, len(ic)):
            glmb_temp_w[ic[hidx]] = glmb_temp_w[ic[hidx]] + self.glmb_update_w[hidx]
            glmb_temp_I[ic[hidx]] = self.glmb_update_I[hidx]
            glmb_temp_n[ic[hidx]] = self.glmb_update_n[hidx]

        self.glmb_update_w = glmb_temp_w  # 2
        self.glmb_update_I = glmb_temp_I  # 3
        self.glmb_update_n = glmb_temp_n  # 4

    def clean_update(self, time_step):
        # flag used tracks
        usedindicator = np.zeros(len(self.glmb_update_tt), dtype=int)
        for hidx in range(0, len(self.glmb_update_w)):
            usedindicator[self.glmb_update_I[hidx]] = usedindicator[self.glmb_update_I[hidx]] + 1
        trackcount = sum(usedindicator > 0)

        # remove unused tracks and reindex existing hypotheses/components
        newindices = np.zeros(len(self.glmb_update_tt), dtype=int)
        newindices[usedindicator > 0] = np.arange(0, trackcount)
        glmb_clean_tt = [self.glmb_update_tt[i] for i, indicator in enumerate(usedindicator) if indicator > 0]

        # remove pruned targets that are kept for 50 frames
        remove_indices = [idx for idx, tt in enumerate(self.prune_glmb_tt) if time_step - tt.last_active > 50]
        remove_indices = sorted(remove_indices, reverse=True)
        for remove_index in remove_indices:
            del self.prune_glmb_tt[remove_index]
            del self.prune_glmb_tt_feat[remove_index]
            del self.prune_glmb_tt_label[remove_index]
        # find pruned targets
        curr_tt_labels = np.unique([t.l for t in glmb_clean_tt])
        pruned_labels = np.setdiff1d(self.prev_tt_glmb_labels, curr_tt_labels)
        if len(pruned_labels):
            for i, tt in enumerate(self.prev_glmb_update_tt):
                if tt.l in pruned_labels:
                    self.prune_glmb_tt.append(tt)
                    self.prune_glmb_tt_feat.append(tt.feat)
                    self.prune_glmb_tt_label.append(tt.l)
        self.prev_tt_glmb_labels = curr_tt_labels

        glmb_clean_I = []
        for hidx in range(0, len(self.glmb_update_w)):
            glmb_clean_I.append(newindices[self.glmb_update_I[hidx]])

        self.glmb_update_tt = glmb_clean_tt  # 1
        self.prev_glmb_update_tt = glmb_clean_tt
        self.glmb_update_I = glmb_clean_I  # 3

    def prune(self):
        # prune components with weights lower than specified threshold
        idxkeep = np.nonzero(self.glmb_update_w > self.model.hyp_threshold)[0]
        glmb_out_w = self.glmb_update_w[idxkeep]
        glmb_out_I = [self.glmb_update_I[i] for i in idxkeep]
        glmb_out_n = self.glmb_update_n[idxkeep]

        glmb_out_w = glmb_out_w / sum(glmb_out_w)
        glmb_out_cdn = np.zeros((max(glmb_out_n) + 1))
        for card in range(0, np.max(glmb_out_n) + 1):
            glmb_out_cdn[card] = sum(glmb_out_w[glmb_out_n == card])

        self.glmb_update_w = glmb_out_w  # 2
        self.glmb_update_I = glmb_out_I  # 3
        self.glmb_update_n = glmb_out_n  # 4
        self.glmb_update_cdn = glmb_out_cdn  # 5

    def cap(self):
        # cap total number of components to specified maximum
        if len(self.glmb_update_w) > self.model.H_max:
            idxsort = np.argsort(-self.glmb_update_w)
            idxkeep = idxsort[0:self.model.H_max]
            # glmb_out.tt= glmb_in.tt;
            glmb_out_w = self.glmb_update_w[idxkeep]
            glmb_out_I = [self.glmb_update_I[i] for i in idxkeep]
            glmb_out_n = self.glmb_update_n[idxkeep]

            glmb_out_w = glmb_out_w / sum(glmb_out_w)
            glmb_out_cdn = np.zeros(max(glmb_out_n) + 1)
            for card in range(0, max(glmb_out_n) + 1):
                glmb_out_cdn[card] = sum(glmb_out_w[glmb_out_n == card])

            self.glmb_update_w = glmb_out_w  # 2
            self.glmb_update_I = glmb_out_I  # 3
            self.glmb_update_n = glmb_out_n  # 4
            self.glmb_update_cdn = glmb_out_cdn  # 5

    def run_glmb(self, z, feat, k):
        # Input z : (n, 4), number of measurements and Top, Left, Bottom, Right of a bounding box
        # Input feat : (n, reid_dim), z[i, :] has re-id feature feat[i, :], norm2 feature
        # tlbr to cxcyah
        z[:, 2:4] -= z[:, 0:2]
        z[:, 0:2] += z[:, 2:4] / 2
        z[:, 2] = z[:, 2] / z[:, 3]
        # joint prediction and update
        self.jointpredictupdate(self.model, z, feat, k)
        # pruning and truncation
        self.prune()
        self.cap()

        # extract estimates via recursive estimator, where trajectories are extracted via association history, and
        # track continuity is guaranteed with a non-trivial estimator

        # extract MAP cardinality and corresponding highest weighted component
        M = np.argmax(self.glmb_update_cdn)
        idxcmp = np.argmax(np.multiply(self.glmb_update_w, (self.glmb_update_n == M).astype(int)))
        X = np.zeros((4, M))
        L = np.zeros(M, dtype=np.dtype('int'))
        for m in range(M):
            # if k > 10 and self.glmb_update_tt[idxptr].estimate < 6:
            #     continue  # ignore false track in GLMB estimation
            idxptr = self.glmb_update_I[idxcmp][m]
            X[:, m] = self.glmb_update_tt[idxptr].m[:4]
            L[m] = self.glmb_update_tt[idxptr].l
        X[2, :] = np.clip(X[2, :], 0.2, None)  # constraint 'a' not to be negative
        if M > 0:
            self.average_area = sum(X[2, :] * X[3, :] ** 2) / M
        return np.copy(X), L
    # END
