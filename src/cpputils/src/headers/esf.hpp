#include <Eigen/SparseCore>

using namespace Eigen;
using namespace  std;

VectorXd esf(VectorXd Z) {
    /*
    Calculate elementary symmetric function using Mahler's recursive formula

    cardinality 1: r1 + r2 + .. + rn
    cardinality 2: r1*r2 + r1*3 + ... + r2*3 + ..

    Parameters
    ----------
    Z: array_like
        Input vector

    Returns
    -------
    out: ndarray
    */
    unsigned int n_z = Z.size();
    VectorXd result(n_z + 1);
    result.setOnes();
    if (n_z == 0) {
        return result;
    }

    MatrixXd F(2, n_z);
    F.setZero();
    unsigned int i_n = 0;
    unsigned int i_n_minus = 1;
    unsigned int n, k, temp;

    for (n = 0; n < n_z; n++) {
        F(i_n, 0) = F(i_n_minus, 0) + Z(n);
        for (k = 1; k < n + 1; k++) {
            if (k == n) {
                F(i_n, k) = Z(n) * F(i_n_minus, k - 1);
            } else {
                F(i_n, k) = F(i_n_minus, k) + Z(n) * F(i_n_minus, k - 1);
            }
        }
        temp = i_n;
        i_n = i_n_minus;
        i_n_minus = temp;
    }
    for (n = 0; n < n_z; n++) {
        result(n + 1) = F(i_n_minus, n);
    }
    return result;
}

double log_sum_exp(double a1, double a2) {
    if (a1 > a2) {
        return a1 + log(exp(a2 - a1) + 1);
    } else {
        return a2 + log(exp(a1 - a2) + 1);
    }
}

VectorXd log_esf(VectorXd Z) {
    /*
    Calculate elementary symmetric function using Mahler's recursive formula
    cardinality 1: log(exp(r1) + exp(r2) + .. + exp(rn))
    cardinality 2: log(exp(r1+r2) + exp(r1+r3) + ... + exp(r2+r3) + ..
    */
    unsigned int n_z = Z.size();
    VectorXd result(n_z + 1);
    result.setZero();
    if (n_z == 0) {
        return result;
    }

    MatrixXd F(2, n_z);
    F.setZero();
    unsigned int i_n = 0;
    unsigned int i_n_minus = 1;
    unsigned int n, k, temp;

    for (n = 0; n < n_z; n++) {
        F(i_n, 0) = log_sum_exp(F(i_n_minus, 0), Z(n));
        for (k = 1; k < n + 1; k++) {
            if (k == n) {
                F(i_n, k) = Z(n) + F(i_n_minus, k - 1);
            }
            else {
                F(i_n, k) = log_sum_exp(F(i_n_minus, k), Z(n) + F(i_n_minus, k - 1));
            }
        }
        temp = i_n;
        i_n = i_n_minus;
        i_n_minus = temp;
    }
    for (n = 0; n < n_z; n++) {
        result(n + 1) = F(i_n_minus, n);
    }
    return result;
}