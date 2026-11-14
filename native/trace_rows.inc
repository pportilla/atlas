// Independent right hand sides of a triangular monomial reduction, modulo
// one trace prime. Each output contains a canonical remainder: pivot entries
// are zero and the surviving entries are in [0,p). This routine also serves
// the CPU-only build and shares no runtime state with device backends.
int trace_rows_cpu(uint32_t p, uint32_t jobs, uint32_t width, const uint32_t* initial,
                   const uint32_t* offsets, const uint32_t* pivots, const uint32_t* targets,
                   const uint32_t* coefficients, uint32_t* output) {
    const u64 square = static_cast<u64>(p) * p;
    // Small jobs are cheaper than constructing a thread crew. Reserve all
    // scratch before launching so worker threads never throw an allocation.
    const uint64_t visits = static_cast<uint64_t>(jobs) * offsets[width];
    const size_t workers = std::max<size_t>(1, std::min<size_t>({jobs, std::thread::hardware_concurrency(),
                                                                 (visits + 262143) / 262144}));
    std::vector<u64> scratch(workers * width);
    std::atomic<uint32_t> next{0};
    auto worker = [&](size_t lane) {
        u64* work = scratch.data() + lane * width;
        for (uint32_t job; (job = next.fetch_add(1, std::memory_order_relaxed)) < jobs;) {
            const size_t at = static_cast<size_t>(job) * width;
            std::copy(initial + at, initial + at + width, work);
            for (uint32_t j = 0; j < width; ++j) {
                const u64 residue = work[j] % p;
                output[at + j] = pivots[j] ? 0 : static_cast<uint32_t>(residue);
                if (!residue || !pivots[j]) continue;
                const u64 factor = p - residue;
                for (uint32_t k = offsets[j]; k < offsets[j + 1]; ++k) {
                    u64& entry = work[targets[k]];
                    entry += factor * coefficients[k];
                    if (entry >= square) entry -= square;
                }
            }
        }
    };
    std::vector<std::thread> crew;
    crew.reserve(workers - 1);
    try {
        for (size_t i = 1; i < workers; ++i) crew.emplace_back(worker, i);
    } catch (const std::system_error&) {
        // The calling thread and any workers already started finish the batch.
    }
    worker(0);
    for (std::thread& t : crew) t.join();
    return 0;
}
