#pragma once
/*
 * MurtyMiller.h
 *
 *  Created on: 15.08.2013
 *      Author: fb
 *
 * Murty's algorithm implementation according to
 * Miller's pseudo-code formulation in "Optimizing Murty's ranked assignment method"
 *
 * Miller, M.L.; Stone, H.S.; Cox, Ingemar J., "Optimizing Murty's ranked assignment method,"
 * Aerospace and Electronic Systems, IEEE Transactions on , vol.33, no.3, pp.851,862, July 1997
 * doi: 10.1109/7.599256
 */

#include "AuctionAlgorithm.hpp"
#include <queue>

using namespace Eigen;

template<typename Scalar = double>
class MurtyMiller {
public:

    typedef Eigen::Matrix<Scalar, -1, -1> WeightMatrix;
    typedef Eigen::Matrix<size_t, -1, -1> AssignmentMatrix;
    typedef typename Auction<Scalar>::Edge Edge;
    typedef typename Auction<Scalar>::Edges Edges;
    typedef std::vector<Edges> Result;

    /**
     * a partition represents an assignment matrix (i.e. edges)
     * with it's weight matrix
     * see Murty's algorithm for details
     */
    class Partition {
    public:
        Partition(const Edges &edges, const WeightMatrix &w, const Scalar v) :
                edges(edges), w(w), value(v) {}

        Edges edges;
        WeightMatrix w;
        Scalar value;
    };

    struct ComparePartition : std::binary_function<Partition, Partition, bool> {
        bool operator()(const Partition &lhs, const Partition &rhs) const {
            return (lhs.value < rhs.value);
        }
    };

    /**
     * list of partitions
     */
    typedef typename std::vector<Partition> Partitions;

    /**
     * sum up values of edges, i.e. objective function value
     * @param edges
     * @return
     */
    static Scalar objectiveFunctionValue(const Edges &edges) {
        Scalar v = 0;
        for (const auto &e : edges)
            v += e.v;

        return v;
    }

    static typename std::tuple<MatrixXi, VectorXd> getKBestAssignments(const WeightMatrix &w, const size_t kBest = 5) {
        const size_t rows = w.rows(), cols = w.cols();

        assert(rows != 0 && cols != 0 && cols >= rows);

        MatrixXi solutions(kBest, rows);
        VectorXd cost(kBest);

        // special case if rows = cols = 1
        if (cols == 1 && rows == 1) {
            if (w(0, 0) == 0) return {MatrixXi(0, 0), VectorXd(0)};

            cost.resize(1);
            cost(0) = w(0, 0);
            return {solutions, cost};
        }

        Edges edges = Auction<Scalar>::solve(w); // make initial (best) assignment

        // sort edges by row
        std::sort(edges.begin(), edges.end(), [](const Edge &e1, const Edge &e2) { return e1.x < e2.x; });

        // initial partition, i.e. best solution
        Partition init(edges, w, objectiveFunctionValue(edges));

        typedef std::priority_queue<Partition, std::vector<Partition>, ComparePartition> PartitionsPriorityQueue;

        // create answer-list with initial partition
        PartitionsPriorityQueue priorityQueue;
        priorityQueue.emplace(init);
        size_t solution_index = 0;

        // assume values between 0 and 1 !
        const Scalar lockingValue = __AUCTION_INF;

        while (!priorityQueue.empty() && solution_index < kBest) {
            // take first element from queue
            Partition currentPartition = priorityQueue.top();
            priorityQueue.pop();

            // for all triplets in this solution
            for (size_t e = 0; e < currentPartition.edges.size(); ++e) {
                auto &triplet = currentPartition.edges[e];

                WeightMatrix P_ = currentPartition.w; // P' = P

                // exclude edge by setting weight in matrix to lockingValue -> NOT (x, y)
                P_(triplet.x, triplet.y) = lockingValue;

                // determine solution for changed matrix and create partition
                Edges S_ = Auction<Scalar>::solve(P_);

                if ((long) S_.size() == P_.rows())// solution found? (rows >= cols!)
                {
                    // sort edges by row
                    std::sort(S_.begin(), S_.end(), [](const Edge &e1, const Edge &e2) { return e1.x < e2.x; });

                    priorityQueue.emplace(Partition(S_, P_, objectiveFunctionValue(S_)));

                }
                // remove all vertices that include row and column of current node
                // i.e. force using this edge
                for (long r = 0; r < currentPartition.w.rows(); ++r)
                    currentPartition.w(r, triplet.y) = lockingValue;

                for (long c = 0; c < currentPartition.w.cols(); ++c)
                    currentPartition.w(triplet.x, c) = lockingValue;

                // set edge back to original value
                currentPartition.w(triplet.x, triplet.y) = triplet.v = w(triplet.x, triplet.y);
            }

            Edges edges_sol = currentPartition.edges;
            for (size_t iidx = 0; iidx < rows; iidx++) {
                solutions(solution_index, iidx) = edges_sol[iidx].y;
            }
            cost(solution_index) = objectiveFunctionValue(edges_sol);
            solution_index += 1;
        }

        return {solutions(seq(0, solution_index - 1), all), cost(seq(0, solution_index - 1))};
    }
};

template<typename Scalar = double>
std::ostream &operator<<(std::ostream &os, const typename MurtyMiller<Scalar>::Result &res) {
    os << "{\n";
    for (auto e: res) { os << "\t" << e << "\n"; }
    os << "}";
    return os;
}
