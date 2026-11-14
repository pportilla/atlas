    // Screen same-degree critical pairs against a frozen basis. The closure
    // of the union of their monomials only needs one reducer per column. Each
    // initial S-polynomial then has its own independent finite-field reduction.
    // Row echelon on the remainders also detects combinations of preceding
    // pairs. The independent pairs go through the ordinary trace in order.
    void screen_degree(size_t budget) {
        const long degree = B.back().sugar;
        size_t first = B.size();
        while (first && B[first - 1].sugar == degree && B.size() - first < 256) --first;
        const size_t jobs = B.size() - first;
        if (jobs < 8) return;
        using Clock = std::chrono::steady_clock;
        Clock::time_point start{}, packed{}, reduced{};
        const TraceProfile profile = trace_profile;
        if (profile) start = Clock::now();
        // These caps limit speculative work, never the underlying exact trace.
        const size_t word_limit = std::min<size_t>(budget, 4u << 20);
        const size_t width_limit = std::min<size_t>(16384, word_limit / jobs);
        begin_block();
        std::vector<size_t> initial_rows;
        initial_rows.reserve(jobs + 1);
        for (size_t i = first; i < B.size(); ++i) {
            const Pair& pair = B[i];
            initial_rows.push_back(rows_.size());
            u64 shift[MAX_STRIDE];
            R.divide(shift, pair.lcm, lead(pair.i));
            add_row(OP_ADD, slot_[pair.i], lead(pair.i), 1, shift);
            R.divide(shift, pair.lcm, lead(pair.j));
            add_row(OP_SUB, slot_[pair.j], lead(pair.j), 1, shift);
            if (pivot_.size() > width_limit || row_terms_.size() > word_limit) return;
        }
        initial_rows.push_back(rows_.size());

        const int stride = R.stride;
        while (!heap_.empty()) {
            std::pop_heap(heap_.begin(), heap_.end(), [this](uint32_t a, uint32_t b) { return smaller(a, b); });
            const uint32_t id = heap_.back();
            heap_.pop_back();
            columns_.push_back(id);
            const u64* mono = arena_.data() + static_cast<size_t>(id) * stride;
            const int g = reducer(mono);
            if (g < 0) continue;
            u64 shift[MAX_STRIDE];
            R.divide(shift, mono, lead(g));
            pivot_[id] = 1;
            add_row(OP_RED, slot_[g], lead(g), 1, shift, id);
            if (pivot_.size() > width_limit || row_terms_.size() > word_limit) return;
        }
        const uint32_t width = static_cast<uint32_t>(columns_.size());
        if (!width) return;
        position_.resize(width);
        for (uint32_t j = 0; j < width; ++j) position_[columns_[j]] = j;

        std::vector<uint32_t> initial(jobs * width);
        for (size_t i = 0; i < jobs; ++i) {
            uint32_t* values = initial.data() + i * width;
            for (size_t r = initial_rows[i]; r < initial_rows[i + 1]; ++r) {
                const Row& row = rows_[r];
                for (uint32_t t = 0; t < row.count; ++t) {
                    uint32_t& value = values[position_[row_terms_[row.terms + t]]];
                    const uint32_t c = pool[row.offset + t];
                    value += row.op == OP_ADD ? c : p_ - c;
                    if (value >= p_) value -= p_;
                }
            }
        }
        std::vector<uint32_t> offsets(width + 1), pivots(width), targets, coefficients;
        const size_t rhs = initial_rows.back();
        const size_t tails = rhs < rows_.size() ? row_terms_.size() - rows_[rhs].terms : 0;
        targets.reserve(tails);
        coefficients.reserve(tails);
        size_t cursor = rhs;
        for (uint32_t col = 0; col < width; ++col) {
            offsets[col] = static_cast<uint32_t>(targets.size());
            if (!pivot_[columns_[col]]) continue;
            pivots[col] = 1;
            const Row& row = rows_[cursor++];
            for (uint32_t t = 0; t < row.count; ++t) {
                targets.push_back(position_[row_terms_[row.terms + t]]);
                coefficients.push_back(pool[row.offset + t]);
            }
        }
        offsets[width] = static_cast<uint32_t>(targets.size());
        std::vector<uint32_t> remainders(initial.size());
        auto reduce = [&](TraceScreen kernel) {
            return kernel(p_, static_cast<uint32_t>(jobs), width, initial.data(), offsets.data(),
                          pivots.data(), targets.data(), coefficients.data(), remainders.data());
        };
        if (profile) packed = Clock::now();
        const bool on_device = trace_screen && reduce(trace_screen) == 0 &&
            std::all_of(remainders.begin(), remainders.end(), [this](uint32_t v) { return v < p_; });
        if (on_device) {
            device_screened = true;
        } else {
            // Overwrite even a partial device result. Reuse the symbolic work
            // already prepared for this layer when no device can run it.
            if (reduce(trace_rows_cpu) != 0) return;
            ++cpu_trace_batches;
            cpu_trace_pairs += jobs;
        }

        if (profile) reduced = Clock::now();

        // Homogeneous pairs of this degree have scalar coefficients in this
        // elimination. B.back() is processed first, in increasing lcm order.
        // A dependent row is a combination of earlier S-polynomials (lcm
        // <= its lcm) and old reducers whose shifted heads are strictly
        // smaller than its lcm. This gives its Buchberger standard
        // representation once those earlier pairs have been accounted for.
        std::vector<int> owner(width, -1);
        std::vector<bool> dependent(jobs);
        std::vector<uint32_t> free_columns;
        for (uint32_t j = 0; j < width; ++j)
            if (!pivots[j]) free_columns.push_back(j);
        for (size_t i = jobs; i-- > 0;) {
            uint32_t* row = remainders.data() + i * width;
            dependent[i] = true;
            for (size_t k = 0; k < free_columns.size(); ++k) {
                const uint32_t j = free_columns[k];
                const uint32_t entry = row[j];
                if (!entry) continue;
                if (owner[j] < 0) {
                    owner[j] = static_cast<int>(i);
                    const u64 inverse = n_invmod(entry, p_);
                    for (size_t t = k + 1; t < free_columns.size(); ++t) {
                        uint32_t& value = row[free_columns[t]];
                        value = static_cast<uint32_t>(inverse * value % p_);
                    }
                    dependent[i] = false;
                    break;
                }
                const uint32_t* source = remainders.data() + static_cast<size_t>(owner[j]) * width;
                const u64 factor = p_ - entry;
                for (size_t t = k + 1; t < free_columns.size(); ++t) {
                    const uint32_t column = free_columns[t];
                    row[column] = static_cast<uint32_t>((row[column] + factor * source[column]) % p_);
                }
            }
        }
        size_t kept = first;
        for (size_t i = 0; i < jobs; ++i) {
            if (dependent[i]) {
                ++reductions;
                ++zero_reductions;
            } else {
                if (kept != first + i) B[kept] = B[first + i];
                ++kept;
            }
        }
        B.resize(kept);
        if (profile) {
            using Seconds = std::chrono::duration<double>;
            profile(static_cast<uint32_t>(degree), p_, static_cast<uint32_t>(jobs), width, offsets[width],
                    static_cast<uint32_t>(kept - first), on_device, Seconds(packed - start).count(),
                    Seconds(reduced - packed).count(), Seconds(Clock::now() - reduced).count());
        }
    }
