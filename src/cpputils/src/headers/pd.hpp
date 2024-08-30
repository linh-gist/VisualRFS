#pragma once

#include "fl/Headers.h"

using namespace fl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
https://stackoverflow.com/questions/59088261/how-to-wrap-a-c-class-to-python-so-that-i-can-access-its-members-public-metho
1) first exporting the Interface
2) override the Interface
*/

class ComputePDInterface {
public:
    virtual ~ComputePDInterface() = default;
};

class ComputePD : public ComputePDInterface {
private:
    Engine *m_engine;
    InputVariable *m_farea_rate;
    InputVariable *m_fioa;
    OutputVariable *m_fpd;

    MatrixXd m_bboxes_ioa;
    VectorXd m_area_rate;
    VectorXd m_avqs;
    VectorXd m_avps;
    MatrixXd m_allcostm;
    double m_model_c;
    double min_pd;
    double max_pd;

public:
    ComputePD() {};

    ComputePD(std::string filename_fis) {
        m_engine = FisImporter().fromFile(filename_fis);
        std::string status;
        if (not m_engine->isReady(&status)) {
            throw Exception("[engine error] engine is not ready:n" + status, FL_AT);
        }
        m_farea_rate = m_engine->getInputVariable("AreaRate");
        m_fioa = m_engine->getInputVariable("IOA");
        m_fpd = m_engine->getOutputVariable("PD");
    }

    double compute(double area_rate_val, double ioa_val) {
        m_farea_rate->setValue(area_rate_val);
        m_fioa->setValue(ioa_val);
        m_engine->process();
        return m_fpd->getValue();
    }

    void set_recompute_cost(VectorXd avqs, VectorXd avps, MatrixXd allcostm, double model_c) {
        m_avqs = avqs;
        m_avps = avps;
        m_allcostm = allcostm;
        m_model_c = model_c;
    }

    double recompute_cost(VectorXd update_hypcmp_tmp, MatrixXd bboxes_ioa, VectorXd area_all) {
        int tt_idx;
        double avpd_new;
        double new_cost = 0;
        int N = update_hypcmp_tmp.size(); // number of tracks
        double area_total = 0;
        int num_survival_track = 0;
        for (tt_idx = 0; tt_idx < N; tt_idx++) {
            if (isinf(update_hypcmp_tmp(tt_idx))) { // target is not born/track deaths
                new_cost -= log(m_avqs(tt_idx));
                bboxes_ioa.col(tt_idx).setZero();
                area_all(tt_idx) = 0;
            } else {
                area_total += area_all(tt_idx);
                num_survival_track += 1;
            }
        }
        area_total /= num_survival_track;
        for (tt_idx = 0; tt_idx < N; tt_idx++) {
            if (isinf(update_hypcmp_tmp(tt_idx))) {
                continue;
            }
            avpd_new = compute(std::min(2.0, area_all(tt_idx) / area_total), bboxes_ioa.row(tt_idx).maxCoeff());
            if (update_hypcmp_tmp(tt_idx) > 0) { // target has an association with a measurement
                new_cost -= log((m_avps(tt_idx) * avpd_new) * m_allcostm(tt_idx, int(update_hypcmp_tmp(tt_idx) - 1)) / m_model_c);
                continue;
            }
            if (update_hypcmp_tmp(tt_idx) == 0) { // target is miss
                new_cost -= log(m_avps(tt_idx) * (1 - avpd_new));
            }
        }
        return new_cost;
    }
};

double recompute_cost(VectorXd hypcmp_tmp, MatrixXd bboxes_ioa, VectorXd avqs, VectorXd avps, MatrixXd allcostm,
                      double model_c, double pd_range[2]) {
    int tt_idx;
    double avpd_new;
    double new_cost = 0;
    int N = hypcmp_tmp.size(); // number of tracks

    for (tt_idx = 0; tt_idx < N; tt_idx++) {
        if (isinf(hypcmp_tmp(tt_idx))) { // target is not born/track deaths
            new_cost -= log(avqs(tt_idx));
            bboxes_ioa.col(tt_idx).setZero();
        }
    }
    for (tt_idx = 0; tt_idx < N; tt_idx++) {
        if (isinf(hypcmp_tmp(tt_idx))) {
            continue;
        }
        avpd_new = std::min(pd_range[1], std::max(pd_range[0], 1 - bboxes_ioa.row(tt_idx).maxCoeff()));
        if (hypcmp_tmp(tt_idx) > 0) { // target has an association with a measurement
            new_cost -= log(
                    (avps(tt_idx) * avpd_new) * allcostm(tt_idx, int(hypcmp_tmp(tt_idx) - 1)) / model_c);
            continue;
        }
        if (hypcmp_tmp(tt_idx) == 0) { // target is miss
            new_cost -= log(avps(tt_idx) * (1 - avpd_new));
        }
    }
    return new_cost;
}