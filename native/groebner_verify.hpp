// Fraction-free certification of a lifted homogeneous basis. Rows of the same
// degree share the monomials and reducer shifts, but have independent exact
// coefficients. The ordinary sparse verifier handles small or very wide cases.
#pragma once

#include "groebner.hpp"

#include <array>
#include <atomic>
#include <exception>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <system_error>
#include <thread>
#include <unordered_map>

namespace atlas {

class HomogeneousCertificate : private Engine {
    using Monomial = std::array<u64, MAX_STRIDE>;
    struct Hash {
        size_t operator()(const Monomial& m) const {
            u64 h = 0;
            for (const u64 word : m) h = (h ^ word) * 0x9E3779B97F4A7C15ULL;
            return static_cast<size_t>(h ^ (h >> 29));
        }
    };
    struct Column {
        int divisor = -1;
        std::vector<uint32_t> tail;
    };
    struct Row {
        std::vector<Z> coefficients;
        std::vector<uint32_t> columns;
    };
    struct Degree {
        std::vector<Column> columns;
        std::vector<uint32_t> order;
        std::vector<Row> rows;
    };
    struct TooWide {};
    // The symbolic closure may densify even when the actual sparse row has
    // cancellations. Bound the total extra graph across degrees, as well as
    // the number of per-thread coefficient slots.
    static constexpr size_t max_columns = 65536, max_edges = 1U << 20;
    std::atomic<size_t> columns_used{0}, edges_used{0};

    void prepare_rows(Degree& plan, std::vector<Poly>& polynomials) {
        std::unordered_map<Monomial, uint32_t, Hash> ids;
        std::vector<Monomial> monomials;
        auto intern = [&](const u64* m) {
            Monomial key{};
            std::copy_n(m, R.stride, key.data());
            const auto entry = ids.emplace(key, static_cast<uint32_t>(monomials.size()));
            if (entry.second) {
                if (columns_used.fetch_add(1, std::memory_order_relaxed) >= max_columns) throw TooWide{};
                monomials.push_back(key);
            }
            return entry.first->second;
        };
        for (Poly& f : polynomials) {
            if (f.empty()) continue;
            Row row;
            row.columns.reserve(f.size());
            for (size_t k = 0; k < f.size(); ++k)
                row.columns.push_back(intern(f.m.data() + k * R.stride));
            row.coefficients = std::move(f.c);
            plan.rows.push_back(std::move(row));
        }
        for (size_t j = 0; j < monomials.size(); ++j) {
            Column col;
            col.divisor = reducer(monomials[j].data());
            if (col.divisor >= 0) {
                const Poly& g = G[col.divisor].p;
                const size_t count = g.size() - 1;
                if (count > max_edges || edges_used.fetch_add(count, std::memory_order_relaxed) > max_edges - count)
                    throw TooWide{};
                col.tail.reserve(g.size() - 1);
                u64 shift[MAX_STRIDE], child[MAX_STRIDE];
                R.divide(shift, monomials[j].data(), g.m.data());
                for (size_t k = 1; k < g.size(); ++k) {
                    R.multiply(child, g.m.data() + k * R.stride, shift);
                    col.tail.push_back(intern(child));
                }
            }
            plan.columns.push_back(std::move(col));
        }
        plan.order.resize(monomials.size());
        std::iota(plan.order.begin(), plan.order.end(), 0);
        std::sort(plan.order.begin(), plan.order.end(), [&](uint32_t a, uint32_t b) {
            return R.compare(monomials[a].data(), monomials[b].data()) > 0;
        });
    }

    bool vanishes(const Degree& plan, const Row& row) const {
        std::vector<Z> work(plan.columns.size());
        for (size_t i = 0; i < row.columns.size(); ++i)
            fmpz_set(&work[row.columns[i]].v, &row.coefficients[i].v);
        Z multiple, scale, gcd;
        unsigned scalings = 0;
        for (auto it = plan.order.begin(); it != plan.order.end(); ++it) {
            const fmpz* pivot = &work[*it].v;
            if (fmpz_is_zero(pivot)) continue;
            const Column& col = plan.columns[*it];
            if (col.divisor < 0) return false;
            const Poly& g = G[col.divisor].p;
            const fmpz* head = &g.c[0].v;
            // Cancel the pivot by scaling the row by head/gcd and subtracting
            // pivot/gcd times the shifted divisor. Its tail is strictly smaller,
            // so a nonzero irreducible column already disproves this certificate.
            if (fmpz_is_one(head)) {
                fmpz_neg(&multiple.v, pivot);
            } else {
                fmpz_gcd(&gcd.v, pivot, head);
                fmpz_divexact(&multiple.v, pivot, &gcd.v);
                fmpz_neg(&multiple.v, &multiple.v);
                fmpz_divexact(&scale.v, head, &gcd.v);
                if (!fmpz_is_one(&scale.v)) {
                    for (auto left = it + 1; left != plan.order.end(); ++left) {
                        auto* a = &work[*left].v;
                        if (!fmpz_is_zero(a)) fmpz_mul(a, a, &scale.v);
                    }
                    ++scalings;
                }
            }
            for (size_t k = 0; k < col.tail.size(); ++k)
                fmpz_addmul(&work[col.tail[k]].v, &multiple.v, &g.c[k + 1].v);
            if (scalings == 8) {
                scalings = 0;
                fmpz_zero(&gcd.v);
                for (auto left = it + 1; left != plan.order.end() && !fmpz_is_one(&gcd.v); ++left)
                    fmpz_gcd(&gcd.v, &gcd.v, &work[*left].v);
                if (!fmpz_is_zero(&gcd.v) && !fmpz_is_one(&gcd.v))
                    for (auto left = it + 1; left != plan.order.end(); ++left) {
                        auto* a = &work[*left].v;
                        if (!fmpz_is_zero(a)) fmpz_divexact(a, a, &gcd.v);
                    }
            }
        }
        return true;
    }

    bool check_rows(const std::map<u64, Degree>& plans) const {
        struct Job { const Degree* degree; const Row* row; };
        std::vector<Job> jobs;
        size_t largest = 1;
        for (const auto& item : plans) {
            const Degree& degree = item.second;
            largest = std::max(largest, degree.columns.size());
            for (const Row& row : degree.rows) jobs.push_back({&degree, &row});
        }
        return all_jobs(jobs.size(), max_columns / largest, [&](size_t k) {
            return vanishes(*jobs[k].degree, *jobs[k].row);
        });
    }

    template<class Operation>
    static bool all_jobs(size_t count, size_t max_workers, Operation run) {
        std::atomic<size_t> cursor{0};
        std::atomic<bool> failed{false};
        std::mutex lock;
        std::exception_ptr error;
        // The workers stop once the caller's deadline passes; see check_deadline.
        const Deadline limit = thread_deadline();
        std::atomic<bool> late{false};
        auto worker = [&] {
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const size_t k = cursor.fetch_add(1, std::memory_order_relaxed);
                    if (k >= count) return;
                    if (passed(limit)) {
                        late.store(true, std::memory_order_relaxed);
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                    if (!run(k)) failed.store(true, std::memory_order_relaxed);
                }
            } catch (...) {
                std::lock_guard<std::mutex> guard(lock);
                if (!error) error = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
            }
        };
        const size_t workers = std::min<size_t>({count, std::max(1U, std::thread::hardware_concurrency()), max_workers});
        std::vector<std::thread> crew;
        crew.reserve(workers);
        try {
            for (size_t i = 1; i < workers; ++i) crew.emplace_back([&] {
                worker();
                flint_cleanup();  // Release this exiting thread's integer cache.
            });
        } catch (const std::system_error&) {
            // If thread creation hits a process limit the caller and any
            // successfully started workers can still exhaust the queue.
        }
        worker();
        for (std::thread& thread : crew) thread.join();
        if (late.load()) expire();
        if (error) std::rethrow_exception(error);
        return !failed;
    }

public:
    // An empty optional asks the caller to perform its ordinary sparse check.
    // Intended for a fresh instance and a homogeneous candidate with the
    // minimal leading monomials from a fully computed trace-prime basis.
    std::optional<bool> check(const Ring& ring, const std::vector<Poly>& candidate,
                              const std::vector<Poly>& generators) {
        if (candidate.size() < 24) return std::nullopt;
        R = ring;
        std::vector<size_t> indices;
        for (size_t k = 0; k < candidate.size(); ++k) {
            const Poly& f = candidate[k];
            if (f.empty()) continue;
            if (fmpz_is_zero(&f.c[0].v)) return false;
            if (f.size() == 1 && f.m[0] == 0) return true;
            indices.push_back(k);
        }
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return R.compare(candidate[a].m.data(), candidate[b].m.data()) < 0;
        });
        for (size_t k : indices) {
            Poly f = candidate[k];
            primitive(f, nullptr);
            insert(std::move(f), degree(candidate[k]));
            if (B.size() > max_columns) return std::nullopt;
        }
        if (B.size() < 64) return std::nullopt;
        std::map<u64, std::vector<Poly>> polynomials;
        size_t seed_terms = 0;
        for (const Pair& pair : B) {
            // spoly reserves the combined tail lengths before cancellation.
            const size_t left = G[pair.i].p.size() - 1, right = G[pair.j].p.size() - 1;
            if (left > max_edges - seed_terms || right > max_edges - seed_terms - left) return std::nullopt;
            seed_terms += left + right;
            Poly f = spoly(pair);
            polynomials[pair.lcm[0]].push_back(std::move(f));
        }
        for (const Poly& f : generators) {
            if (f.empty()) continue;
            if (f.size() > max_edges - seed_terms) return std::nullopt;
            seed_terms += f.size();
            polynomials[f.m[0]].push_back(f);
        }
        std::map<u64, Degree> plans;
        try {
            // Each degree owns its symbolic closure; only the overall count
            // of allocated columns and tail entries is shared.
            using Group = std::pair<Degree*, std::vector<Poly>*>;
            std::vector<Group> groups;
            for (auto& group : polynomials) groups.emplace_back(&plans[group.first], &group.second);
            std::reverse(groups.begin(), groups.end());
            all_jobs(groups.size(), groups.size(), [&](size_t k) {
                prepare_rows(*groups[k].first, *groups[k].second);
                return true;
            });
        } catch (const TooWide&) {
            return std::nullopt;
        }
        return check_rows(plans);
    }
};

}  // namespace atlas
