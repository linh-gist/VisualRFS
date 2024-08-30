#include <vector>
#include <math.h>       /* exp */
#include <random>		/*uniform distribution*/
#include <tuple>		/* return assignments & costs*/
#include <Eigen/Core>
#include "lap.hpp"

using namespace Eigen;
using namespace  std;

std::tuple<MatrixXd, VectorXd> gibbs_jointpredupdt(MatrixXd P0, int m) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    uniform_real_distribution<double> distribution(0.0, 1.0);

    size_t p0_row = P0.rows();
    size_t p0_col = P0.cols();

    if (m == 0) {
        m = 1;
    }
    MatrixXd P0_exp = (-1 * P0).array().exp();
    vector<vector<int>> assignments(m, vector<int>(p0_row));
    VectorXd tempsamp(p0_col);
    vector<double> idxold(p0_col + 1, 0);
    vector<double> cumsum(p0_col + 1, 0.0);

    // use LAPJV output as initial solution
    lap::Slack u(p0_row);
    lap::Slack v(p0_col);
    lap::Assignment x(p0_row);
    lap::lap(P0.array() - P0.minCoeff(), x, u, v);
    vector<int> currsoln(x.data(), x.data() + x.size());
    assignments[0] = currsoln;
    for (int sol = 1; sol < m; sol++) {
        for (int var = 0; var < p0_row; var++) {
            // grab row of costs for current association variable
            tempsamp = P0_exp.row(var);
            // lock out current and previous iteration step assignments except for the one in question
            for (int i = 0; i < p0_row; i++) {
                if (i == var)
                    continue;
                tempsamp[currsoln[i]] = 0.0;
            }
            int idx = 1;
            for (int i = 0; i < p0_col; i++) {
                if (tempsamp[i] > 0) {
                    idxold[idx] = i;
                    cumsum[idx] = cumsum[idx - 1] + tempsamp[i];
                    idx++;
                }
            }
            double sum_tempsamp = cumsum[idx - 1];
            double rand_num = distribution(generator);
            for (int i = 1; i < idx; i++) {
                if ((cumsum[i] / sum_tempsamp) > rand_num) {
                    currsoln[var] = i;
                    break;
                }
            }
            currsoln[var] = idxold[currsoln[var]];
        }
        assignments[sol] = currsoln;
    }
    std::sort(assignments.begin(), assignments.end());
    assignments.erase(std::unique(assignments.begin(), assignments.end()), assignments.end());

    // calculate costs for each assignment
    int costs_size = assignments.size();
    MatrixXi assignments_xd(costs_size, p0_row);
    VectorXd costs(costs_size);
    costs.setZero();
    for (int i = 0; i < costs_size; i++) {
        assignments_xd.row(i) = VectorXi::Map(&assignments[i][0], p0_row);
        for (int j = 0; j < p0_row; j++) {
            costs(i) += P0(j, assignments[i][j]);
        }
    }

    return std::make_tuple(assignments_xd.cast<double>(), costs);
}