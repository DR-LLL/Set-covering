#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

struct SetCoverInstance {
    int n;
    int m;
    std::vector<double> costs;
    std::vector<std::vector<int> > sets;
    std::vector<std::vector<int> > elem_to_sets;
    std::vector<int> set_ids;

    static SetCoverInstance from_file(const std::string& path) {
        std::ifstream fin(path.c_str());
        if (!fin) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        std::string header;
        if (!std::getline(fin, header)) {
            throw std::runtime_error("Empty file (missing header 'n m').");
        }

        std::istringstream hs(header);
        int n, m;
        if (!(hs >> n >> m)) {
            throw std::runtime_error("Header must be 'n m'.");
        }

        if (n <= 0 || m <= 0) {
            throw std::runtime_error("n and m must be positive.");
        }

        SetCoverInstance inst;
        inst.n = n;
        inst.m = m;
        inst.costs.reserve(m);
        inst.sets.reserve(m);
        inst.elem_to_sets.assign(n, std::vector<int>());
        inst.set_ids.resize(m);

        for (int i = 0; i < m; ++i) {
            std::string line;
            if (!std::getline(fin, line)) {
                throw std::runtime_error("Unexpected EOF: expected more set lines.");
            }

            std::istringstream ls(line);
            double c;
            if (!(ls >> c)) {
                throw std::runtime_error("Bad set line: missing cost.");
            }

            std::set<int> unique_elems;
            int e;
            while (ls >> e) {
                if (e < 0 || e >= n) {
                    throw std::runtime_error("Element index out of range.");
                }
                unique_elems.insert(e);
            }

            std::vector<int> elems(unique_elems.begin(), unique_elems.end());
            inst.costs.push_back(c);
            inst.sets.push_back(elems);
            inst.set_ids[i] = i;
        }

        for (int si = 0; si < m; ++si) {
            for (size_t k = 0; k < inst.sets[si].size(); ++k) {
                int e = inst.sets[si][k];
                inst.elem_to_sets[e].push_back(si);
            }
        }

        return inst;
    }
};

struct SetCoverSolution {
    std::vector<int> chosen_set_ids;
    double total_cost;
    bool is_feasible;
    int uncovered_count;
    double best_found_time;

    SetCoverSolution()
        : total_cost(std::numeric_limits<double>::infinity()),
          is_feasible(false),
          uncovered_count(0),
          best_found_time(0.0) {}
};

struct CostDescendingComparator {
    const SetCoverInstance* inst;

    explicit CostDescendingComparator(const SetCoverInstance* instance) : inst(instance) {}

    bool operator()(int a, int b) const {
        if (inst->costs[a] != inst->costs[b]) {
            return inst->costs[a] > inst->costs[b];
        }
        return a < b;
    }
};

class SetCoverSolver {
public:
    SetCoverSolver(
        const SetCoverInstance& instance,
        double time_limit_seconds,
        unsigned int base_seed,
        double beta_start = 1.0,
        double beta_min = 0.8,
        double beta_decay = 0.995,
        int grasp_period = 10,
        double keep_fraction = 0.8
    )
        : instance_(instance),
          time_limit_seconds_(time_limit_seconds),
          beta_start_(beta_start),
          beta_min_(beta_min),
          beta_decay_(beta_decay),
          grasp_period_(grasp_period),
          keep_fraction_(keep_fraction),
          base_seed_(base_seed),
          rng_(base_seed) {
        if (time_limit_seconds_ <= 0.0) {
            throw std::runtime_error("time_limit_seconds must be > 0");
        }
        if (beta_start_ <= 0.0 || beta_start_ > 1.0) {
            throw std::runtime_error("beta_start must be in (0, 1].");
        }
        if (beta_min_ <= 0.0 || beta_min_ > 1.0) {
            throw std::runtime_error("beta_min must be in (0, 1].");
        }
        if (beta_min_ > beta_start_) {
            throw std::runtime_error("beta_min must be <= beta_start.");
        }
        if (beta_decay_ <= 0.0 || beta_decay_ > 1.0) {
            throw std::runtime_error("beta_decay must be in (0, 1].");
        }
        if (grasp_period_ <= 0) {
            throw std::runtime_error("grasp_period must be > 0");
        }
        if (keep_fraction_ < 0.0 || keep_fraction_ > 1.0) {
            throw std::runtime_error("keep_fraction must be in [0, 1].");
        }
    }

    SetCoverSolution solve() {
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

        SetCoverSolution initial = greedy_from_scratch_with_forbidden(std::vector<char>(instance_.m, 0));
        SetCoverSolution best_solution = local_search(initial);

        if (!best_solution.is_feasible) {
            best_solution.best_found_time = elapsed_seconds(t0);
            return best_solution;
        }

        best_solution.best_found_time = elapsed_seconds(t0);

        int iteration = 0;
        double current_beta = beta_start_;

        while (elapsed_seconds(t0) < time_limit_seconds_) {
            ++iteration;

            bool grasp_step = (iteration % grasp_period_ == 0);
            SetCoverSolution current;

            if (grasp_step) {
                int keep_count = compute_keep_count(best_solution);
                double random_grasp_alpha = random_unit_double();
                current = grasp_repair_from_best(best_solution, keep_count, random_grasp_alpha);
            } else {
                std::vector<char> forbidden(instance_.m, 0);
                build_random_forbidden_from_best(best_solution, forbidden, current_beta);
                current = repair_from_best_with_forbidden(best_solution, forbidden);
            }

            if (!current.is_feasible) {
                current_beta = next_beta(current_beta);
                continue;
            }

            current = local_search(current);
            if (!current.is_feasible) {
                current_beta = next_beta(current_beta);
                continue;
            }

            if (strictly_better_objective(current, best_solution)) {
                best_solution = current;
                best_solution.best_found_time = elapsed_seconds(t0);
                current_beta = beta_start_;

                std::cout
                    << "new best at iter=" << iteration
                    << " cost=" << best_solution.total_cost
                    << " chosen_sets=" << best_solution.chosen_set_ids.size()
                    << " best_found_time=" << std::fixed << std::setprecision(6)
                    << best_solution.best_found_time << "s\n";
            } else if (same_objective(current, best_solution)) {
                double old_best_time = best_solution.best_found_time;
                best_solution = current;
                best_solution.best_found_time = old_best_time;
                current_beta = beta_start_;
            } else {
                current_beta = next_beta(current_beta);
            }
        }

        return best_solution;
    }

private:
    const SetCoverInstance& instance_;
    double time_limit_seconds_;
    double beta_start_;
    double beta_min_;
    double beta_decay_;
    int grasp_period_;
    double keep_fraction_;
    unsigned int base_seed_;
    std::mt19937 rng_;

    double elapsed_seconds(const std::chrono::steady_clock::time_point& t0) const {
        return std::chrono::duration_cast<std::chrono::duration<double> >(
            std::chrono::steady_clock::now() - t0
        ).count();
    }

    double next_beta(double current_beta) const {
        double next = current_beta * beta_decay_;
        if (next < beta_min_) {
            next = beta_min_;
        }
        return next;
    }

    double random_unit_double() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng_);
    }

    bool strictly_better_objective(const SetCoverSolution& a, const SetCoverSolution& b) const {
        if (!a.is_feasible && !b.is_feasible) {
            return false;
        }
        if (a.is_feasible && !b.is_feasible) {
            return true;
        }
        if (!a.is_feasible && b.is_feasible) {
            return false;
        }
        return a.total_cost < b.total_cost;
    }

    bool same_objective(const SetCoverSolution& a, const SetCoverSolution& b) const {
        if (!a.is_feasible || !b.is_feasible) {
            return false;
        }
        return a.total_cost == b.total_cost;
    }

    bool is_better_for_local_search(const SetCoverSolution& a, const SetCoverSolution& b) const {
        if (!a.is_feasible && !b.is_feasible) {
            return false;
        }
        if (a.is_feasible && !b.is_feasible) {
            return true;
        }
        if (!a.is_feasible && b.is_feasible) {
            return false;
        }
        if (a.total_cost < b.total_cost) {
            return true;
        }
        if (a.total_cost > b.total_cost) {
            return false;
        }
        return a.chosen_set_ids.size() < b.chosen_set_ids.size();
    }

    bool set_covers_element(int set_id, int element) const {
        const std::vector<int>& elems = instance_.sets[set_id];
        return std::binary_search(elems.begin(), elems.end(), element);
    }

    int sample_geometric_destroy_size(double beta) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double u = dist(rng_);

        if (beta >= 1.0) {
            return 1;
        }

        double value = std::log(1.0 - u) / std::log(1.0 - beta);
        int x = static_cast<int>(value) + 1;
        if (x < 1) {
            x = 1;
        }
        return x;
    }

    void build_random_forbidden_from_best(
        const SetCoverSolution& best_solution,
        std::vector<char>& forbidden,
        double beta
    ) {
        int k = static_cast<int>(best_solution.chosen_set_ids.size());
        if (k == 0) {
            return;
        }

        int destroy_size = sample_geometric_destroy_size(beta);
        if (destroy_size > k) {
            destroy_size = k;
        }

        std::vector<int> selected = best_solution.chosen_set_ids;
        std::shuffle(selected.begin(), selected.end(), rng_);

        for (int i = 0; i < destroy_size; ++i) {
            forbidden[selected[i]] = 1;
        }
    }

    int compute_keep_count(const SetCoverSolution& best_solution) const {
        int k = static_cast<int>(best_solution.chosen_set_ids.size());
        if (k <= 0) {
            return 0;
        }

        int keep_count = static_cast<int>(std::floor(keep_fraction_ * static_cast<double>(k)));
        if (keep_count < 0) {
            keep_count = 0;
        }
        if (keep_count > k) {
            keep_count = k;
        }
        return keep_count;
    }

    int build_rcl_and_choose(const std::vector<std::pair<int, double> >& candidate_scores, double alpha) {
        if (candidate_scores.empty()) {
            return -1;
        }

        double max_score = candidate_scores[0].second;
        double min_score = candidate_scores[0].second;

        for (size_t i = 1; i < candidate_scores.size(); ++i) {
            if (candidate_scores[i].second > max_score) {
                max_score = candidate_scores[i].second;
            }
            if (candidate_scores[i].second < min_score) {
                min_score = candidate_scores[i].second;
            }
        }

        double threshold = max_score - alpha * (max_score - min_score);

        std::vector<int> rcl;
        for (size_t i = 0; i < candidate_scores.size(); ++i) {
            if (candidate_scores[i].second >= threshold) {
                rcl.push_back(candidate_scores[i].first);
            }
        }

        if (rcl.empty()) {
            return candidate_scores[0].first;
        }

        std::uniform_int_distribution<int> pick_dist(0, static_cast<int>(rcl.size()) - 1);
        return rcl[pick_dist(rng_)];
    }

    SetCoverSolution greedy_from_scratch_with_forbidden(const std::vector<char>& forbidden) {
        const int n = instance_.n;
        const int m = instance_.m;

        std::vector<char> uncovered(n, 1);
        int uncovered_count = n;

        std::vector<int> chosen_local;
        std::vector<char> chosen_flag(m, 0);
        double total_cost = 0.0;

        while (uncovered_count > 0) {
            int best_i = -1;

            for (int e = 0; e < n; ++e) {
                if (!uncovered[e]) {
                    continue;
                }

                int candidate_count = 0;
                int last_candidate = -1;

                const std::vector<int>& covering_sets = instance_.elem_to_sets[e];
                for (size_t k = 0; k < covering_sets.size(); ++k) {
                    int si = covering_sets[k];
                    if (forbidden[si] || chosen_flag[si]) {
                        continue;
                    }
                    ++candidate_count;
                    last_candidate = si;
                    if (candidate_count > 1) {
                        break;
                    }
                }

                if (candidate_count == 0) {
                    SetCoverSolution sol;
                    sol.chosen_set_ids = chosen_local;
                    sol.total_cost = total_cost;
                    sol.is_feasible = false;
                    sol.uncovered_count = uncovered_count;
                    return sol;
                }

                if (candidate_count == 1) {
                    best_i = last_candidate;
                    break;
                }
            }

            if (best_i == -1) {
                double best_score = -1.0;

                for (int i = 0; i < m; ++i) {
                    if (forbidden[i] || chosen_flag[i]) {
                        continue;
                    }

                    int gain = 0;
                    const std::vector<int>& elems = instance_.sets[i];
                    for (size_t k = 0; k < elems.size(); ++k) {
                        if (uncovered[elems[k]]) {
                            ++gain;
                        }
                    }

                    if (gain == 0) {
                        continue;
                    }

                    double c = instance_.costs[i];
                    if (c <= 0.0) {
                        best_i = i;
                        best_score = std::numeric_limits<double>::infinity();
                        break;
                    }

                    double score = static_cast<double>(gain) / c;
                    if (score > best_score) {
                        best_score = score;
                        best_i = i;
                    }
                }

                if (best_i == -1) {
                    SetCoverSolution sol;
                    sol.chosen_set_ids = chosen_local;
                    sol.total_cost = total_cost;
                    sol.is_feasible = false;
                    sol.uncovered_count = uncovered_count;
                    return sol;
                }
            }

            chosen_flag[best_i] = 1;
            chosen_local.push_back(best_i);
            total_cost += instance_.costs[best_i];

            const std::vector<int>& elems = instance_.sets[best_i];
            for (size_t k = 0; k < elems.size(); ++k) {
                int e = elems[k];
                if (uncovered[e]) {
                    uncovered[e] = 0;
                    --uncovered_count;
                }
            }
        }

        SetCoverSolution sol;
        sol.chosen_set_ids = chosen_local;
        sol.total_cost = total_cost;
        sol.is_feasible = true;
        sol.uncovered_count = 0;
        return sol;
    }

    SetCoverSolution repair_from_best_with_forbidden(
        const SetCoverSolution& best_solution,
        const std::vector<char>& forbidden
    ) {
        const int n = instance_.n;
        const int m = instance_.m;

        std::vector<char> chosen_flag(m, 0);
        std::vector<int> coverage_count(n, 0);
        std::vector<char> uncovered(n, 1);
        std::vector<int> chosen_local;
        double total_cost = 0.0;

        for (size_t i = 0; i < best_solution.chosen_set_ids.size(); ++i) {
            int sid = best_solution.chosen_set_ids[i];
            if (forbidden[sid]) {
                continue;
            }

            chosen_flag[sid] = 1;
            chosen_local.push_back(sid);
            total_cost += instance_.costs[sid];

            const std::vector<int>& elems = instance_.sets[sid];
            for (size_t k = 0; k < elems.size(); ++k) {
                ++coverage_count[elems[k]];
            }
        }

        int uncovered_count = 0;
        for (int e = 0; e < n; ++e) {
            if (coverage_count[e] > 0) {
                uncovered[e] = 0;
            } else {
                uncovered[e] = 1;
                ++uncovered_count;
            }
        }

        while (uncovered_count > 0) {
            int best_i = -1;

            for (int e = 0; e < n; ++e) {
                if (!uncovered[e]) {
                    continue;
                }

                int candidate_count = 0;
                int last_candidate = -1;

                const std::vector<int>& covering_sets = instance_.elem_to_sets[e];
                for (size_t k = 0; k < covering_sets.size(); ++k) {
                    int si = covering_sets[k];
                    if (forbidden[si] || chosen_flag[si]) {
                        continue;
                    }
                    ++candidate_count;
                    last_candidate = si;
                    if (candidate_count > 1) {
                        break;
                    }
                }

                if (candidate_count == 0) {
                    SetCoverSolution sol;
                    sol.chosen_set_ids = chosen_local;
                    sol.total_cost = total_cost;
                    sol.is_feasible = false;
                    sol.uncovered_count = uncovered_count;
                    return sol;
                }

                if (candidate_count == 1) {
                    best_i = last_candidate;
                    break;
                }
            }

            if (best_i == -1) {
                double best_score = -1.0;

                for (int i = 0; i < m; ++i) {
                    if (forbidden[i] || chosen_flag[i]) {
                        continue;
                    }

                    int gain = 0;
                    const std::vector<int>& elems = instance_.sets[i];
                    for (size_t k = 0; k < elems.size(); ++k) {
                        if (uncovered[elems[k]]) {
                            ++gain;
                        }
                    }

                    if (gain == 0) {
                        continue;
                    }

                    double c = instance_.costs[i];
                    if (c <= 0.0) {
                        best_i = i;
                        best_score = std::numeric_limits<double>::infinity();
                        break;
                    }

                    double score = static_cast<double>(gain) / c;
                    if (score > best_score) {
                        best_score = score;
                        best_i = i;
                    }
                }

                if (best_i == -1) {
                    SetCoverSolution sol;
                    sol.chosen_set_ids = chosen_local;
                    sol.total_cost = total_cost;
                    sol.is_feasible = false;
                    sol.uncovered_count = uncovered_count;
                    return sol;
                }
            }

            chosen_flag[best_i] = 1;
            chosen_local.push_back(best_i);
            total_cost += instance_.costs[best_i];

            const std::vector<int>& elems = instance_.sets[best_i];
            for (size_t k = 0; k < elems.size(); ++k) {
                int e = elems[k];
                ++coverage_count[e];
                if (uncovered[e]) {
                    uncovered[e] = 0;
                    --uncovered_count;
                }
            }
        }

        std::sort(chosen_local.begin(), chosen_local.end());

        SetCoverSolution sol;
        sol.chosen_set_ids = chosen_local;
        sol.total_cost = total_cost;
        sol.is_feasible = true;
        sol.uncovered_count = 0;
        return sol;
    }

    SetCoverSolution grasp_repair_from_best(
        const SetCoverSolution& best_solution,
        int keep_count,
        double alpha
    ) {
        const int n = instance_.n;
        const int m = instance_.m;

        std::vector<char> chosen_flag(m, 0);
        std::vector<int> coverage_count(n, 0);
        std::vector<char> uncovered(n, 1);
        std::vector<int> chosen_local;
        double total_cost = 0.0;

        std::vector<int> selected = best_solution.chosen_set_ids;
        std::shuffle(selected.begin(), selected.end(), rng_);

        if (keep_count > static_cast<int>(selected.size())) {
            keep_count = static_cast<int>(selected.size());
        }
        if (keep_count < 0) {
            keep_count = 0;
        }

        for (int i = 0; i < keep_count; ++i) {
            int sid = selected[i];
            chosen_flag[sid] = 1;
            chosen_local.push_back(sid);
            total_cost += instance_.costs[sid];

            const std::vector<int>& elems = instance_.sets[sid];
            for (size_t k = 0; k < elems.size(); ++k) {
                ++coverage_count[elems[k]];
            }
        }

        int uncovered_count = 0;
        for (int e = 0; e < n; ++e) {
            if (coverage_count[e] > 0) {
                uncovered[e] = 0;
            } else {
                uncovered[e] = 1;
                ++uncovered_count;
            }
        }

        while (uncovered_count > 0) {
            int best_i = -1;

            for (int e = 0; e < n; ++e) {
                if (!uncovered[e]) {
                    continue;
                }

                int candidate_count = 0;
                int last_candidate = -1;

                const std::vector<int>& covering_sets = instance_.elem_to_sets[e];
                for (size_t k = 0; k < covering_sets.size(); ++k) {
                    int si = covering_sets[k];
                    if (chosen_flag[si]) {
                        continue;
                    }
                    ++candidate_count;
                    last_candidate = si;
                    if (candidate_count > 1) {
                        break;
                    }
                }

                if (candidate_count == 0) {
                    SetCoverSolution sol;
                    sol.chosen_set_ids = chosen_local;
                    sol.total_cost = total_cost;
                    sol.is_feasible = false;
                    sol.uncovered_count = uncovered_count;
                    return sol;
                }

                if (candidate_count == 1) {
                    best_i = last_candidate;
                    break;
                }
            }

            if (best_i == -1) {
                std::vector<std::pair<int, double> > candidate_scores;
                candidate_scores.reserve(m);

                for (int i = 0; i < m; ++i) {
                    if (chosen_flag[i]) {
                        continue;
                    }

                    int gain = 0;
                    const std::vector<int>& elems = instance_.sets[i];
                    for (size_t k = 0; k < elems.size(); ++k) {
                        if (uncovered[elems[k]]) {
                            ++gain;
                        }
                    }

                    if (gain == 0) {
                        continue;
                    }

                    double c = instance_.costs[i];
                    double score;
                    if (c <= 0.0) {
                        score = std::numeric_limits<double>::infinity();
                    } else {
                        score = static_cast<double>(gain) / c;
                    }

                    candidate_scores.push_back(std::make_pair(i, score));
                }

                if (candidate_scores.empty()) {
                    SetCoverSolution sol;
                    sol.chosen_set_ids = chosen_local;
                    sol.total_cost = total_cost;
                    sol.is_feasible = false;
                    sol.uncovered_count = uncovered_count;
                    return sol;
                }

                best_i = build_rcl_and_choose(candidate_scores, alpha);
                if (best_i == -1) {
                    SetCoverSolution sol;
                    sol.chosen_set_ids = chosen_local;
                    sol.total_cost = total_cost;
                    sol.is_feasible = false;
                    sol.uncovered_count = uncovered_count;
                    return sol;
                }
            }

            chosen_flag[best_i] = 1;
            chosen_local.push_back(best_i);
            total_cost += instance_.costs[best_i];

            const std::vector<int>& elems = instance_.sets[best_i];
            for (size_t k = 0; k < elems.size(); ++k) {
                int e = elems[k];
                ++coverage_count[e];
                if (uncovered[e]) {
                    uncovered[e] = 0;
                    --uncovered_count;
                }
            }
        }

        std::sort(chosen_local.begin(), chosen_local.end());

        SetCoverSolution sol;
        sol.chosen_set_ids = chosen_local;
        sol.total_cost = total_cost;
        sol.is_feasible = true;
        sol.uncovered_count = 0;
        return sol;
    }

    SetCoverSolution redundancy_elimination(const SetCoverSolution& solution) {
        if (!solution.is_feasible) {
            return solution;
        }

        std::vector<int> chosen = solution.chosen_set_ids;
        std::vector<int> coverage_count(instance_.n, 0);

        for (size_t i = 0; i < chosen.size(); ++i) {
            int sid = chosen[i];
            const std::vector<int>& elems = instance_.sets[sid];
            for (size_t k = 0; k < elems.size(); ++k) {
                ++coverage_count[elems[k]];
            }
        }

        std::sort(chosen.begin(), chosen.end(), CostDescendingComparator(&instance_));

        std::vector<int> kept_sets;
        kept_sets.reserve(chosen.size());

        for (size_t i = 0; i < chosen.size(); ++i) {
            int sid = chosen[i];
            bool removable = true;

            const std::vector<int>& elems = instance_.sets[sid];
            for (size_t k = 0; k < elems.size(); ++k) {
                if (coverage_count[elems[k]] <= 1) {
                    removable = false;
                    break;
                }
            }

            if (removable) {
                for (size_t k = 0; k < elems.size(); ++k) {
                    --coverage_count[elems[k]];
                }
            } else {
                kept_sets.push_back(sid);
            }
        }

        std::sort(kept_sets.begin(), kept_sets.end());

        double total_cost = 0.0;
        for (size_t i = 0; i < kept_sets.size(); ++i) {
            total_cost += instance_.costs[kept_sets[i]];
        }

        int uncovered_count = 0;
        for (int e = 0; e < instance_.n; ++e) {
            if (coverage_count[e] <= 0) {
                ++uncovered_count;
            }
        }

        SetCoverSolution improved;
        improved.chosen_set_ids = kept_sets;
        improved.total_cost = total_cost;
        improved.is_feasible = (uncovered_count == 0);
        improved.uncovered_count = uncovered_count;
        return improved;
    }

    SetCoverSolution exchange_improvement(const SetCoverSolution& solution) {
        if (!solution.is_feasible) {
            return solution;
        }

        std::vector<int> chosen = solution.chosen_set_ids;
        std::vector<char> chosen_flag(instance_.m, 0);
        std::vector<int> coverage_count(instance_.n, 0);

        for (size_t i = 0; i < chosen.size(); ++i) {
            chosen_flag[chosen[i]] = 1;
        }

        for (size_t i = 0; i < chosen.size(); ++i) {
            int sid = chosen[i];
            const std::vector<int>& elems = instance_.sets[sid];
            for (size_t k = 0; k < elems.size(); ++k) {
                ++coverage_count[elems[k]];
            }
        }

        bool improved = true;

        while (improved) {
            improved = false;
            std::sort(chosen.begin(), chosen.end(), CostDescendingComparator(&instance_));

            for (size_t idx = 0; idx < chosen.size() && !improved; ++idx) {
                int sid_remove = chosen[idx];
                double remove_cost = instance_.costs[sid_remove];

                std::vector<int> critical_elements;
                const std::vector<int>& remove_elems = instance_.sets[sid_remove];

                for (size_t k = 0; k < remove_elems.size(); ++k) {
                    int e = remove_elems[k];
                    if (coverage_count[e] == 1) {
                        critical_elements.push_back(e);
                    }
                }

                if (critical_elements.empty()) {
                    for (size_t k = 0; k < remove_elems.size(); ++k) {
                        --coverage_count[remove_elems[k]];
                    }
                    chosen_flag[sid_remove] = 0;
                    chosen.erase(chosen.begin() + idx);
                    improved = true;
                    break;
                }

                int best_add = -1;
                double best_add_cost = std::numeric_limits<double>::infinity();

                for (int sid_add = 0; sid_add < instance_.m; ++sid_add) {
                    if (chosen_flag[sid_add]) {
                        continue;
                    }

                    double add_cost = instance_.costs[sid_add];
                    if (add_cost >= remove_cost) {
                        continue;
                    }

                    bool covers_all = true;
                    for (size_t ce = 0; ce < critical_elements.size(); ++ce) {
                        if (!set_covers_element(sid_add, critical_elements[ce])) {
                            covers_all = false;
                            break;
                        }
                    }

                    if (covers_all && add_cost < best_add_cost) {
                        best_add = sid_add;
                        best_add_cost = add_cost;
                    }
                }

                if (best_add != -1) {
                    for (size_t k = 0; k < remove_elems.size(); ++k) {
                        --coverage_count[remove_elems[k]];
                    }
                    chosen_flag[sid_remove] = 0;
                    chosen.erase(chosen.begin() + idx);

                    const std::vector<int>& add_elems = instance_.sets[best_add];
                    for (size_t k = 0; k < add_elems.size(); ++k) {
                        ++coverage_count[add_elems[k]];
                    }
                    chosen_flag[best_add] = 1;
                    chosen.push_back(best_add);

                    improved = true;
                    break;
                }
            }
        }

        std::sort(chosen.begin(), chosen.end());

        double total_cost = 0.0;
        for (size_t i = 0; i < chosen.size(); ++i) {
            total_cost += instance_.costs[chosen[i]];
        }

        int uncovered_count = 0;
        for (int e = 0; e < instance_.n; ++e) {
            if (coverage_count[e] <= 0) {
                ++uncovered_count;
            }
        }

        SetCoverSolution improved_solution;
        improved_solution.chosen_set_ids = chosen;
        improved_solution.total_cost = total_cost;
        improved_solution.is_feasible = (uncovered_count == 0);
        improved_solution.uncovered_count = uncovered_count;
        return improved_solution;
    }

    SetCoverSolution local_search(const SetCoverSolution& solution) {
        if (!solution.is_feasible) {
            return solution;
        }

        SetCoverSolution current = solution;
        bool improved = true;

        while (improved) {
            improved = false;

            SetCoverSolution after_red = redundancy_elimination(current);
            if (is_better_for_local_search(after_red, current)) {
                current = after_red;
                continue;
            }

            SetCoverSolution after_swap = exchange_improvement(current);
            if (is_better_for_local_search(after_swap, current)) {
                current = after_swap;
                continue;
            }
        }

        return current;
    }
};

static bool is_regular_file(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    return S_ISREG(st.st_mode);
}

static bool is_directory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    return S_ISDIR(st.st_mode);
}

static std::string join_solution_ids(const std::vector<int>& ids) {
    std::ostringstream oss;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            oss << ' ';
        }
        oss << ids[i];
    }
    return oss.str();
}

void run_all_tests_in_folder(
    const std::string& folder,
    double time_limit_seconds,
    const std::string& results_csv,
    unsigned int base_seed,
    double beta_start = 1.0,
    double beta_min = 0.8,
    double beta_decay = 0.995,
    int grasp_period = 10,
    double keep_fraction = 0.8
) {
    if (!is_directory(folder)) {
        throw std::runtime_error("Folder does not exist: " + folder);
    }

    DIR* dir = opendir(folder.c_str());
    if (!dir) {
        throw std::runtime_error("Cannot open folder: " + folder);
    }

    std::vector<std::string> files;
    struct dirent* entry;

    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }

        std::string full_path = folder + "/" + name;
        if (is_regular_file(full_path)) {
            files.push_back(name);
        }
    }
    closedir(dir);

    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cout << "No files found in folder: " << folder << "\n";
        return;
    }

    std::ofstream fout(results_csv.c_str());
    if (!fout) {
        throw std::runtime_error("Cannot open CSV for writing: " + results_csv);
    }

    fout << "test_name,objective_value,best_found_time_sec,solution\n";

    std::cout << "Folder: " << folder << "\n";
    std::cout << "Files: " << files.size() << "\n";
    std::cout << "Per-test time limit: " << std::fixed << std::setprecision(3) << time_limit_seconds << "s\n";
    std::cout << "Beta start: " << std::fixed << std::setprecision(3) << beta_start << "\n";
    std::cout << "Beta min: " << std::fixed << std::setprecision(3) << beta_min << "\n";
    std::cout << "Beta decay: " << std::fixed << std::setprecision(3) << beta_decay << "\n";
    std::cout << "D (GRASP partial period): " << grasp_period << "\n";
    std::cout << "t (keep fraction): " << std::fixed << std::setprecision(3) << keep_fraction << "\n";
    std::cout << "Base seed: " << base_seed << "\n";
    std::cout << "CSV output: " << results_csv << "\n\n";

    for (size_t i = 0; i < files.size(); ++i) {
        const std::string& fn = files[i];
        const std::string path = folder + "/" + fn;

        try {
            SetCoverInstance inst = SetCoverInstance::from_file(path);

            // Для каждого теста RNG начинается заново с одного и того же base_seed
            SetCoverSolver solver(
                inst,
                time_limit_seconds,
                base_seed,
                beta_start,
                beta_min,
                beta_decay,
                grasp_period,
                keep_fraction
            );

            SetCoverSolution sol = solver.solve();

            if (sol.is_feasible && std::isfinite(sol.total_cost)) {
                fout << fn << ","
                     << std::setprecision(15) << sol.total_cost << ","
                     << std::setprecision(15) << sol.best_found_time << ","
                     << "\"" << join_solution_ids(sol.chosen_set_ids) << "\"\n";
            } else {
                fout << fn << ",ERROR,ERROR,\"\"\n";
            }
        } catch (const std::exception&) {
            fout << fn << ",ERROR,ERROR,\"\"\n";
        }
    }
}

static unsigned int generate_random_seed() {
    std::random_device rd;
    unsigned long long t = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    unsigned long long r1 = static_cast<unsigned long long>(rd());
    unsigned long long r2 = static_cast<unsigned long long>(rd());
    unsigned long long mixed = t ^ (r1 << 1) ^ (r2 << 33);
    return static_cast<unsigned int>(mixed & 0xffffffffu);
}

int main() {
    try {
        const unsigned int base_seed = 4040336150;

        run_all_tests_in_folder(
            "data",
            60,
            "results.csv",
            base_seed,
            1.0,
            0.8,
            0.995,
            4,
            0.8
        );
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}