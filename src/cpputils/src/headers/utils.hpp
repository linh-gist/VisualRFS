//
// Created by Linh Ma (linh.mavan@gm.gist.ac.kr) on 21. 11. 17..
//

#pragma once

#include <Eigen/SparseCore>

using namespace Eigen;

double recompute_cost(VectorXd hypcmp_tmp, MatrixXd bboxes_ioi, VectorXd avqs, VectorXd avps, MatrixXd allcostm,
                      double model_c) {
    int tt_idx;
    double avpd_new;
    double new_cost = 0;
    int N = hypcmp_tmp.size(); // number of tracks

    for (tt_idx = 0; tt_idx < N; tt_idx++) {
        if (isinf(hypcmp_tmp(tt_idx))) { // target is not born/track deaths
            new_cost -= log(avqs(tt_idx));
            bboxes_ioi.col(tt_idx).setZero();
        }
    }
    for (tt_idx = 0; tt_idx < N; tt_idx++) {
        if (isinf(hypcmp_tmp(tt_idx))) {
            continue;
        }
        avpd_new = std::min(0.95, std::max(0.7, bboxes_ioi.row(tt_idx).maxCoeff()));
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

double log_sum_exp(VectorXd arr) {
    int count = arr.size();
    if (count > 0) {
        double maxVal = arr.maxCoeff();
        double sum = 0;
        for (int i = 0; i < count; i++) {
            sum += exp(arr(i) - maxVal);
        }
        return log(sum) + maxVal;
    } else {
        return 0.0;
    }
}

Eigen::VectorXi select_nonnegative(Eigen::VectorXd x){
    std::vector<int> result;
    for (int i = 0; i < x.size(); i++) {
        if (x(i) >= 0) {
            result.push_back(x(i));
        }
    }
    return Eigen::VectorXi::Map(result.data(), result.size());
}

VectorXd bbox_iou_xyah_eigen(VectorXd box, std::vector<VectorXd> query_boxes) {
	/*
	Parameters
	----------
	box: (4) array of float
	query_boxes: (K, 4) 2d array of float
	Returns
	-------
	overlaps: (K) array of overlap between box and query_boxes
	*/
	unsigned int K = query_boxes.size();
	VectorXd overlaps(K);
	double iw, ih, box_area, query_area, ua;
	unsigned int k, n;
	double query_t, query_l, query_b, query_r, query_w;
	double box_t, box_l, box_b, box_r, box_w;
	box_w = box(2) * box(3);
	box_t = box(1) - box(3) / 2;  // 0
	box_l = box(0) - box_w / 2;   // 1
	box_b = box(1) + box(3) / 2;  // 2
	box_r = box(0) + box_w / 2;   // 3
	box_area = (box_w + 1) * (box(3) + 1);
	for (k = 0; k < K; k++) {
		query_w = query_boxes[k](2) * query_boxes[k](3);
		query_t = query_boxes[k](1) - query_boxes[k](3) / 2;  // 0
		query_l = query_boxes[k](0) - query_w / 2;                  // 1
		query_b = query_boxes[k](1) + query_boxes[k](3) / 2;  // 2
		query_r = query_boxes[k](0) + query_w / 2;                  // 3
		query_area = (query_w + 1) *  (query_boxes[k](3) + 1);
		iw = (
			std::min(box_b, query_b) -
			std::max(box_t, query_t) + 1
			);
		if (iw > 0) {
			ih = (
				std::min(box_r, query_r) -
				std::max(box_l, query_l) + 1
				);
			if (ih > 0) {
				ua = box_area + query_area - iw * ih;
				overlaps(k) = iw * ih / ua;
			}
		}
	}
	return overlaps;
}