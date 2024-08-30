//
// Created by Linh Ma (linh.mavan@gm.gist.ac.kr) on 21. 11. 17..
//

#ifndef JOINT_GLMB_MODEL_HPP
#define JOINT_GLMB_MODEL_HPP

#include <Eigen/SparseCore>

class Model{
public:
    int H_upd;
    int H_max;
    double hyp_threshold;
    double gamma;
    double lambda_c;
    double model_c;
    Eigen::MatrixXd R;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd F;
    Eigen::MatrixXd H;
    double b_thresh;
    double lambda_b;
    double prob_birth;

    Model() {
        H_upd = 500; // requested number of updated components/hypotheses
        H_max = 500;  // cap on number of posterior components/hypotheses
        hyp_threshold = 1e-10;  // pruning threshold for components/hypotheses

        int z_dim = 4;
        double P_G = 0.95;  // gate size in percentage
        gamma = 9.487729036781154; // chi2.ppf(P_G, z_dim)  # inv chi^2 dn gamma value
        double P_D = .8; // probability of detection in measurements
        double P_S = .99; // survival/death parameters

        // clutter parameters
        lambda_c = 0.5;  // poisson average rate of uniform clutter (per scan)
        double range_c = 1920 * 1080; //np.array([[0, 1920], [0, 1080]])  # uniform clutter region
        double pdf_c = 1 / range_c; //np.prod(self.range_c[:, 1] - self.range_c[:, 0])  # uniform clutter density
        model_c = lambda_c * pdf_c;

        // observation noise covariance
        R.resize(4,4); Q.resize(8,8); F.resize(8,8); H.resize(4,8);
        R <<
            70., 0., 0., 0.,
            0., 30., 0., 0.,
            0., 0., 0.0055, 0.,
            0., 0., 0., 70.;
        float T = 1;  // sate vector [x,y,a,h,dx,dy,da,dh]
        double sigma_xy = 25; double sigma_a = 1e-4; double sigma_h = 25;
        Q <<
            pow(T,4) * (sigma_xy / 4), 0, 0, 0, pow(T,3) * (sigma_xy / 2), 0, 0, 0,  // process noise covariance
            0, pow(T,4) * (sigma_xy / 4), 0, 0, 0, pow(T,3) * (sigma_xy / 2), 0, 0,
            0, 0, pow(T,4) * (sigma_a / 4), 0, 0, 0, pow(T,3) * (sigma_a / 2), 0,
            0, 0, 0, pow(T,4) * (sigma_h / 4), 0, 0, 0, pow(T,3) * (sigma_h / 2),
            pow(T,3) * (sigma_xy / 2), 0, 0, 0, sigma_xy * pow(T,2), 0, 0, 0,
            0, pow(T,3)* (sigma_xy / 2), 0, 0, 0, sigma_xy * pow(T,2), 0, 0,
            0, 0, pow(T,3) * (sigma_a / 2), 0, 0, 0, sigma_a * pow(T,2), 0,
            0, 0, 0, pow(T,3) * (sigma_h / 2), 0, 0, 0, sigma_h * pow(T,2);

        F <<
            1, 0, 0, 0, 1, 0, 0, 0,  // Motion model: state transition matrix
            0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 1;
        H <<
        1, 0, 0, 0, 0, 0, 0, 0,  // observation matrix
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0;

        b_thresh = 0.95;  // only birth a new target at a measurement that has lower assign_prob than this threshold
        lambda_b = 0.1;  // Set lambda_b to the mean cardinality of the birth multi-Bernoulli RFS
        prob_birth = 5e-4;  // Initial existence probability of a birth track
        /* NOTE: prob_birth=0.03, lambda_b=0.1 need to be chosen carefully so that tt.r is not too small (e.g. 5e-4)
           NOTE: init covariance, P=diag(50) and lamda_c=0.5 are not too high (e.g. P=diag(1000), lamda_c=8)
           NOTE: P: high => uncertainty high => qz low, tt.r small => cannot create birth tracks */
    }
};

#endif //JOINT_GLMB_MODEL_HPP