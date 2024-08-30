#include <omp.h>
#include <vector>
#include <math.h>       /* exp */
#include <random>		/*uniform distribution*/
#include <tuple>		/* return assignments & costs*/
#include <algorithm>
#include <Eigen/SparseCore>

using namespace  std;

Eigen::VectorXd bboxes_ioi_xyah_back2front(Eigen::MatrixXd tt_lmb, Eigen::VectorXi tt_lmb_l, Eigen::MatrixXd t_x, Eigen::VectorXi tt_x_l){
    /*
	Parameters
    ----------
    tt_lmb (N, 4) & t_x: (M, 4) ndarray of float
    Returns
    -------
    overlaps: (N) array of mutual overlap between boxes from back to front
    */
	unsigned int N = tt_lmb.rows();
	unsigned int M = t_x.rows();
	Eigen::VectorXd overlaps(N);
	double iw, ih, query_area, ioi, ioi_temp;
	unsigned int k, n;
	double query_t, query_l, query_b, query_r, query_w;
	double box_t, box_l, box_b, box_r, box_w;
	for (k = 0; k < N; k++){
		query_w = tt_lmb(k, 2) * tt_lmb(k, 3);
		query_t = tt_lmb(k, 1) - tt_lmb(k, 3) / 2;       // 0
		query_l = tt_lmb(k, 0) - query_w / 2;       // 1
		query_b = tt_lmb(k, 1) + tt_lmb(k, 3) / 2;       // 2
		query_r = tt_lmb(k, 0) + query_w / 2;       // 3
		query_area = (
			(query_w + 1) *     // (query_b - query_t + 1)
			(tt_lmb(k, 3) + 1)  // (query_r - query_l + 1)
			);
		ioi = 0;
		for (n = 0; n < M; n++){
			box_b = t_x(n, 1) + t_x(n, 3) / 2;      // 2

			if ((box_b < query_b) || (tt_lmb_l[k] == tt_x_l[n])) {
				continue; // ignore objects stand behind or itself
			}
			box_w = t_x(n, 2) * t_x(n, 3);
			box_t = t_x(n, 1) - t_x(n, 3) / 2;      // 0
			box_l = t_x(n, 0) - box_w / 2;          // 1
			box_r = t_x(n, 0) + box_w / 2;          // 3
			iw = (
				std::min(box_b, query_b) -
				std::max(box_t, query_t) + 1
				);
            if (iw > 0) {
				ih = (
					std::min(box_r, query_r) -
					std::max(box_l, query_l) + 1
					);
                if (ih > 0){
					ioi_temp = iw * ih / query_area;
                    if (ioi < ioi_temp){
						ioi = ioi_temp;
					}
				}
			}
		}
		overlaps[k] = ioi;
	}
	return overlaps;
}

Eigen::MatrixXd bboxes_ioi_xyah_back2front_all(Eigen::MatrixXd tt_xyah) {
    /*
    Parameters
    ----------
    tt_lmb (N, 4) ndarray of float
    Returns
    -------
    overlaps: (N, N) ndarray of mutual overlap between boxes from back to front
    */
    unsigned int N = tt_xyah.rows();
    double iw, ih, query_area, ioi, ioi_temp;
    unsigned int k, n;
    Eigen::VectorXd tmp_width, tmp_height;

    Eigen::MatrixXd overlaps(N, N);
    overlaps.setZero();
    if (N == 0) {
        return overlaps;
    }

    tt_xyah.col(2) = tt_xyah(Eigen::all, {2,3}).rowwise().prod(); // width
    tmp_width = Eigen::Map<Eigen::VectorXd>(tt_xyah.col(2).data(), N);
    tmp_height = Eigen::Map<Eigen::VectorXd>(tt_xyah.col(3).data(), N);

    tt_xyah.col(1) -= tt_xyah.col(3) / 2;  // top
    tt_xyah.col(0) -= tt_xyah.col(2) / 2;  // left
    tt_xyah.col(3) += tt_xyah.col(1);      // bottom
    tt_xyah.col(2) += tt_xyah.col(0);      // right
    for (k = 0; k < N; k++) {
        query_area = ((tmp_width[k] + 1) * (tmp_height[k] + 1));
        ioi = 0;
        for (n = 0; n < N; n++) {
            if ((tt_xyah(n, 3) < tt_xyah(k, 3)) || (n == k)) {
                continue; // ignore objects stand behind or itself
            }
            iw = (
                    std::min(tt_xyah(n, 2), tt_xyah(k, 2)) -
                    std::max(tt_xyah(n, 0), tt_xyah(k, 0)) + 1
            );
            if (iw > 0) {
                ih = (
                        std::min(tt_xyah(n, 3), tt_xyah(k, 3)) -
                        std::max(tt_xyah(n, 1), tt_xyah(k, 1)) + 1
                );
                if (ih > 0) {
                    overlaps(k, n) = iw * ih / query_area;
                }
            }
        }
    }
    return overlaps;
}

Eigen::VectorXd bbox_iou_xyah(Eigen::VectorXd box, Eigen::MatrixXd query_boxes) {
	/*
	Parameters
	----------
	box: (4) array of float
	query_boxes: (K, 4) 2d array of float
	Returns
	-------
	overlaps: (K) array of overlap between box and query_boxes
	*/
	unsigned int K = query_boxes.rows();
	Eigen::VectorXd overlaps(K);
	overlaps.setZero();
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
		query_w = query_boxes(k, 2) * query_boxes(k, 3);
		query_t = query_boxes(k, 1) - query_boxes(k, 3) / 2;  // 0
		query_l = query_boxes(k, 0) - query_w / 2;            // 1
		query_b = query_boxes(k, 1) + query_boxes(k, 3) / 2;  // 2
		query_r = query_boxes(k, 0) + query_w / 2;            // 3
		query_area = (query_w + 1) *  (query_boxes(k, 3) + 1);
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