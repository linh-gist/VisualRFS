import numpy as np
from cpputils import Murty
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal


def kalman_predict_single(F, Q, m, P):
    m_predict = np.dot(F, m)
    P_predict = Q + np.dot(F, np.dot(P, F.T))
    return m_predict, P_predict


def gate_meas_gms_idx(z, feat, model, m, P, tt_feat):
    zlength = z.shape[0]
    if zlength == 0:
        return np.empty(0)
    # similar reid features have cosine distance less than 0.3
    cdist_gate = cdist(feat, tt_feat[np.newaxis, :], metric='cosine')[:, 0] < 0.3

    Sj = model.R + np.dot(np.dot(model.H, P), model.H.T)
    Vs = np.linalg.cholesky(Sj)
    inv_sqrt_Sj = np.linalg.inv(Vs)
    nu = z - np.tile(np.dot(model.H, m), (zlength, 1))
    dist = sum(np.square(np.dot(inv_sqrt_Sj, nu.T)))
    valid_idx_tmp = np.nonzero(np.logical_or(dist < model.gamma, cdist_gate))[0]

    return valid_idx_tmp

def kalman_update_single(z, H, R, m, P):
    mu = np.dot(H, m)
    S = R + np.dot(np.dot(H, P), H.T)
    Vs = np.linalg.cholesky(S);
    inv_sqrt_S = np.linalg.inv(Vs);
    iS = np.dot(inv_sqrt_S, inv_sqrt_S.T)
    K = np.dot(np.dot(P, H.T), iS)

    z_mu = z - mu
    qz_temp = multivariate_normal.pdf(z, mean=mu, cov=S)
    m_temp = m + np.dot(K, z_mu)
    P_temp = np.dot((np.eye(len(P)) - np.dot(K, H)), P)

    return qz_temp, m_temp, P_temp

def kalman_predict_multiple(model, m, P):
    plength = m.shape[1];

    m_predict = np.zeros(m.shape);
    P_predict = np.zeros(P.shape);

    for idxp in range(0, plength):
        m_temp, P_temp = kalman_predict_single(model.F, model.Q, m[:, idxp], P[:, :, idxp]);
        m_predict[:, idxp] = m_temp;
        P_predict[:, :, idxp] = P_temp;

    return m_predict, P_predict

def kalman_update_multiple(z, model, m, P):
    plength = m.shape[1];
    zlength = z.shape[1];

    qz_update = np.zeros(plength, zlength);
    m_update = np.zeros(model.x_dim, plength, zlength);
    P_update = np.zeros(model.x_dim, model.x_dim, plength);

    for idxp in range(0, plength):
        qz_temp, m_temp, P_temp = kalman_update_single(z, model.H, model.R, m[:, idxp], P[:, :, idxp]);
        qz_update[idxp, :] = qz_temp;
        m_update[:, idxp, :] = m_temp;
        P_update[:, :, idxp] = P_temp;

    return qz_update, m_update, P_update

def sub2ind(array_shape, rows, cols):
    return rows + array_shape[0] * cols

def unique_faster(keys):
    keys = np.sort(keys)
    difference = np.diff(np.append(keys, np.nan))
    keys = keys[np.nonzero(difference)[0]]

    return keys

def gibbswrap_jointpredupdt_custom(P0, m):
    n1 = P0.shape[0];

    if m == 0:
        m = 1 # return at least one solution

    assignments = np.zeros((m, n1));
    costs = np.zeros(m);

    currsoln = np.arange(n1, 2 * n1);  # use all missed detections as initial solution
    assignments[0, :] = currsoln;
    costs[0] = sum(P0.flatten('F')[sub2ind(P0.shape, np.arange(0, n1), currsoln)]);
    for sol in range(1, m):
        for var in range(0, n1):
            tempsamp = np.exp(-P0[var, :]);  # grab row of costs for current association variable
            # lock out current and previous iteration step assignments except for the one in question
            tempsamp[np.delete(currsoln, var)] = 0;
            idxold = np.nonzero(tempsamp > 0)[0];
            tempsamp = tempsamp[idxold];
            currsoln[var] = np.digitize(np.random.rand(1), np.concatenate(([0], np.cumsum(tempsamp) / sum(tempsamp))));
            currsoln[var] = idxold[currsoln[var]-1];
        assignments[sol, :] = currsoln;
        costs[sol] = sum(P0.flatten('F')[sub2ind(P0.shape, np.arange(0, n1), currsoln)]);
    C, I, _ = np.unique(assignments, return_index=True, return_inverse=True, axis=0);
    assignments = C;
    costs = costs[I];

    return assignments, costs

def murty(P0, m):
    n1 = P0.shape[0]
    if n1 == 0:
        return np.empty(0), np.empty(0)
    mgen = Murty(P0)
    assignments = np.zeros((m, n1))
    costs = np.zeros(m)
    sol_idx = 0
    # for cost, assignment in murty(C_ext):
    for sol in range(0, m):
        ok, cost_m, assignment_m = mgen.draw()
        if (not ok):
            break
        assignments[sol, :] = assignment_m
        costs[sol] = cost_m
        sol_idx += 1
    C, I, _ = np.unique(assignments[:sol_idx, :], return_index=True, return_inverse=True, axis=0)
    assignments = C
    costs = costs[I]
    return assignments, costs

if __name__ == '__main__':
    P0 = np.array([[0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, -0.345108847352739, np.inf, np.inf],
                   [np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, -0.849090957754662, np.inf],
                   [np.inf, np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, 1.64038243547480],
                   [np.inf, np.inf, np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf]])
    gibbswrap_jointpredupdt_custom(P0, 1000)