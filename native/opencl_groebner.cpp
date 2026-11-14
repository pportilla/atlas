// Optional OpenCL 1.2 plugin for the modular Groebner route (modular.cpp):
// it runs one replay program for a batch of primes, one work item per prime.
// Every work item executes the same instructions on its own coefficients, so
// the items never diverge. The CPU library does not link against OpenCL; the
// host passes atlas_replay_opencl to atlas_groebner_use_batch.
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif
#include "trace_levels.hpp"

#include <algorithm>
#ifdef ATLAS_OPENCL_PROFILE
#include <chrono>
#endif
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

// The instruction set of modular.cpp. Work entries stay below p*p < 2^62.
const char* source = R"CLC(
// A reduced monomial basis has no tails. Each term can give one divisor
// index (or -1 when irreducible) independently of all the other terms.
kernel void monomial_divisors(global const int *powers, global const int *basis,
                              int variables, int divisors, global int *witnesses) {
    const size_t row = get_global_id(0);
    int witness = -1;
    for (int j = 0; j < divisors && witness < 0; ++j) {
        int k = 0;
        while (k < variables && basis[j * variables + k] <= powers[row * variables + k]) ++k;
        if (k == variables) witness = j;
    }
    witnesses[row] = witness;
}

// One independent S-pair per lane. Neighbouring lanes access the same
// monomial in different pairs. Reducer tails have strictly later columns.
kernel void trace_screen(uint prime, uint pairs, uint width,
                         global const uint *initial, global const uint *offsets,
                         global const uint *pivot, global const uint *columns,
                         global const uint *coefficients, global ulong *work, global uint *remainders) {
    size_t item = get_global_id(0);
    for (uint j = 0; j < width; ++j)
        work[(size_t)j * pairs + item] = initial[item * width + j];
    const ulong square = (ulong)prime * prime;
    for (uint j = 0; j < width; ++j) {
        ulong value = work[(size_t)j * pairs + item];
        uint f = prime == 2147483629u ? (uint)(value % 2147483629ul) : (uint)(value % prime);
        remainders[item * width + j] = pivot[j] ? 0 : f;
        if (!f || !pivot[j]) continue;
        const ulong factor = prime - f;
        for (uint t = offsets[j]; t < offsets[j+1]; ++t) {
            size_t at = (size_t)columns[t] * pairs + item;
            ulong x = work[at] + factor * coefficients[t];
            work[at] = x >= square ? x - square : x;
        }
    }
}

// Each workgroup owns an independent row. Four lanes share the incoming
// edges of a column. Destinations of the same level have no edges between
// them, so the full group can advance several columns together.
kernel void trace_levels(uint prime, uint pairs, uint width,
                         global const uint *initial, global const uint *starts,
                         global const uint *pivot, global const uint *parents,
                         global const uint *coefficients, local ulong *storage,
                         global uint *out, global const uint *order,
                         global const uint *levels, uint depth) {
    const uint lane = get_local_id(0), group = get_local_size(0);
    const uint team = lane / 4, member = lane % 4;
    const size_t row = get_group_id(0) * width;
    const ulong square = (ulong)prime * prime;
    local uint *reduced = (local uint *)storage;
    local ulong *partial = storage + (width + 1) / 2;
    for (uint d = 0; d < depth; ++d) {
        for (uint base = levels[d]; base < levels[d+1]; base += group / 4) {
            const uint i = base + team;
            const uint col = i < levels[d+1] ? order[i] : 0;
            ulong sum = 0;
            if (i < levels[d+1]) {
                for (uint t = starts[col] + member; t < starts[col+1]; t += 4) {
                    sum += (ulong)reduced[parents[t]] * coefficients[t];
                    if (sum >= square) sum -= square;
                }
            }
            partial[lane] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (member == 0 && i < levels[d+1]) {
                ulong value = initial[row+col];
                for (uint q = 0; q < 4; ++q) {
                    value += partial[lane+q];
                    if (value >= square) value -= square;
                }
                const uint f = prime == 2147483629u ? (uint)(value % 2147483629ul) : (uint)(value % prime);
                reduced[col] = prime - f;
                out[row+col] = pivot[col] ? 0 : f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

kernel void replay(global const uint *code, uint from, uint to, uint pool_size, uint work_size,
                   global const uint *primes, global uint *pools, global ulong *works, global uint *status) {
    const size_t item = get_global_id(0);
    if (!status[item]) return;
    const uint p = primes[item];
    const ulong square = (ulong)p * p;
    global uint *pool = pools + item * (size_t)pool_size;
    global ulong *work = works + item * (size_t)work_size;
    uint at = from;
    while (at < to) {
        const uint op = code[at];
        if (op == 1) {
            const uint width = code[at + 1];
            for (uint t = 0; t < width; ++t) work[t] = 0;
            at += 2;
        } else if (op == 2 || op == 3) {
            global const uint *src = pool + code[at + 1];
            const uint count = code[at + 2];
            global const uint *index = code + at + 3;
            for (uint t = 0; t < count; ++t) {
                ulong w = work[index[t]] + (op == 2 ? src[t] : p - src[t]);
                work[index[t]] = w >= square ? w - square : w;
            }
            at += 3 + count;
        } else if (op == 4) {
            const uint column = code[at + 1];
            global const uint *src = pool + code[at + 2];
            const uint count = code[at + 3];
            global const uint *index = code + at + 4;
            const ulong f = work[column] % p;
            work[column] = 0;
            if (f) {
                const ulong factor = p - f;
                for (uint t = 0; t < count; ++t) {
                    ulong w = work[index[t]] + factor * src[t];
                    work[index[t]] = w >= square ? w - square : w;
                }
            }
            at += 4 + count;
        } else if (op == 5) {
            const uint count = code[at + 1];
            global const uint *index = code + at + 2;
            for (uint t = 0; t < count; ++t)
                if (work[index[t]] % p) { status[item] = 0; return; }
            at += 2 + count;
        } else if (op == 6) {
            global uint *target = pool + code[at + 1];
            const uint count = code[at + 2];
            global const uint *index = code + at + 3;
            ulong base = work[index[0]] % p;
            if (!base) { status[item] = 0; return; }
            ulong inverse = 1;
            for (uint e = p - 2; e; e >>= 1) {
                if (e & 1) inverse = inverse * base % p;
                base = base * base % p;
            }
            target[0] = 1;
            for (uint t = 1; t < count; ++t) target[t] = (uint)((work[index[t]] % p) * inverse % p);
            at += 3 + count;
        } else {
            status[item] = 0;
            return;
        }
    }
}
)CLC";

void check(cl_int status) {
    if (status != CL_SUCCESS) throw std::runtime_error("OpenCL call failed");
}

template <typename Handle, cl_int (*Release)(Handle)> struct Owned {
    Handle value = nullptr;
    ~Owned() { if (value) Release(value); }
    Owned() = default;
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
};
using Buffer = Owned<cl_mem, clReleaseMemObject>;

#ifdef ATLAS_OPENCL_PROFILE
using Clock = std::chrono::steady_clock;
using Event = Owned<cl_event, clReleaseEvent>;
thread_local double trace_times[5];
double event_ms(cl_event event) {
    cl_ulong start, end;
    check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr));
    check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
    return static_cast<double>(end - start) * 1e-6;
}
#endif

// The probe and the executor must agree on the indices of all GPU devices.
// An enumeration error on one platform leaves absence of a given index unknown.
std::vector<cl_device_id> algebra_devices(bool& complete) {
    complete = true;
    cl_uint count = 0;
    cl_int status = clGetPlatformIDs(0, nullptr, &count);
#ifdef CL_PLATFORM_NOT_FOUND_KHR
    if (status == CL_PLATFORM_NOT_FOUND_KHR) return {};
#endif
    check(status);
    if (!count) return {};
    std::vector<cl_platform_id> platforms(count);
    check(clGetPlatformIDs(count, platforms.data(), nullptr));
    std::vector<cl_device_id> devices;
    for (auto platform : platforms) {
        cl_uint n = 0;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &n);
        if (status == CL_DEVICE_NOT_FOUND) continue;
        if (status != CL_SUCCESS) {
            complete = false;
            continue;
        }
        if (!n) continue;
        std::vector<cl_device_id> ids(n);
        check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, n, ids.data(), nullptr));
        devices.insert(devices.end(), ids.begin(), ids.end());
    }
    return devices;
}

struct Runtime {
    Owned<cl_context, clReleaseContext> context;
    Owned<cl_program, clReleaseProgram> program;
    Owned<cl_command_queue, clReleaseCommandQueue> queue;
    Owned<cl_kernel, clReleaseKernel> kernel;
    Owned<cl_kernel, clReleaseKernel> screen;
    Owned<cl_kernel, clReleaseKernel> screen_levels;
    Owned<cl_kernel, clReleaseKernel> monomial_divisors;
    cl_ulong local_bytes = 0;
    size_t screen_group = 1;
    cl_device_id device = nullptr;

    explicit Runtime(size_t index) {
        bool complete;
        auto found = algebra_devices(complete);
        if (index >= found.size()) throw std::runtime_error("No such OpenCL GPU");
        device = found[index];
        cl_int status;
        context.value = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        check(status);
        program.value = clCreateProgramWithSource(context.value, 1, &source, nullptr, &status);
        check(status);
        check(clBuildProgram(program.value, 1, &device, "-cl-std=CL1.2", nullptr, nullptr));
        cl_command_queue_properties properties = 0;
#ifdef ATLAS_OPENCL_PROFILE
        properties = CL_QUEUE_PROFILING_ENABLE;
#endif
        queue.value = clCreateCommandQueue(context.value, device, properties, &status);
        check(status);
        kernel.value = clCreateKernel(program.value, "replay", &status);
        check(status);
        screen.value = clCreateKernel(program.value, "trace_screen", &status);
        check(status);
        screen_levels.value = clCreateKernel(program.value, "trace_levels", &status);
        check(status);
        monomial_divisors.value = clCreateKernel(program.value, "monomial_divisors", &status);
        check(status);
        check(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_bytes), &local_bytes, nullptr));
        size_t max_group;
        check(clGetKernelWorkGroupInfo(screen_levels.value, device, CL_KERNEL_WORK_GROUP_SIZE,
                                       sizeof(max_group), &max_group, nullptr));
        // Teams contain four lanes, independently of a device's subgroup size.
        screen_group = std::min<size_t>(64, max_group) / 4 * 4;
    }
};

std::mutex lock;  // one batch at a time: the kernel object holds its arguments

Runtime& runtime(size_t index) {
    static std::map<size_t, std::unique_ptr<Runtime>> instances;
    auto found = instances.find(index);
    if (found != instances.end()) return *found->second;
    auto instance = std::make_unique<Runtime>(index);
    Runtime& selected = *instance;
    instances.emplace(index, std::move(instance));
    return selected;
}

size_t default_device() {
    const char* wanted = std::getenv("RESOLUTION_GPU_INDEX");
    return wanted ? std::strtoul(wanted, nullptr, 10) : 0;
}

template <typename T> void arg(cl_kernel kernel, cl_uint index, T value) {
    check(clSetKernelArg(kernel, index, sizeof(T), &value));
}

}  // namespace

extern "C" {

// 1 = present, 0 = absent, -1 = indeterminate. No kernel compilation.
int atlas_groebner_opencl_device_status(size_t index) noexcept {
    try {
        bool complete;
        auto found = algebra_devices(complete);
        if (index < found.size()) return 1;
        return complete ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

int atlas_monomial_divisors_opencl_index(const int32_t* powers, const int32_t* basis,
        uint32_t terms, uint32_t divisors, uint32_t variables, int32_t* witnesses, size_t index) noexcept {
    // Python bounds both the packed upload and the number of coordinate comparisons.
    if (!terms || !variables || variables > 32 || !divisors ||
        uint64_t(terms) * divisors > (1u << 24) / variables) return 1;
    try {
        std::lock_guard<std::mutex> guard(lock);
        Runtime& rt = runtime(index);
        cl_kernel k = rt.monomial_divisors.value;
        cl_int status;
        Buffer terms_buf, basis_buf, result;
        terms_buf.value = clCreateBuffer(rt.context.value, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(cl_int)*terms*variables, const_cast<int32_t*>(powers), &status);
        check(status);
        basis_buf.value = clCreateBuffer(rt.context.value, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(cl_int)*divisors*variables, const_cast<int32_t*>(basis), &status);
        check(status);
        result.value = clCreateBuffer(rt.context.value, CL_MEM_WRITE_ONLY, sizeof(cl_int)*terms, nullptr, &status);
        check(status);
        arg(k, 0, terms_buf.value);
        arg(k, 1, basis_buf.value);
        arg(k, 2, static_cast<cl_int>(variables));
        arg(k, 3, static_cast<cl_int>(divisors));
        arg(k, 4, result.value);
        const size_t global = terms;
        check(clEnqueueNDRangeKernel(rt.queue.value, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr));
        check(clEnqueueReadBuffer(rt.queue.value, result.value, CL_TRUE, 0, sizeof(cl_int)*terms,
                                  witnesses, 0, nullptr, nullptr));
        return 0;
    } catch (...) {
        return 1;
    }
}

#ifdef ATLAS_OPENCL_PROFILE
// Timing builds synchronize after the kernel to separate execution from readback.
// Wall times in milliseconds for packing/allocation/upload, launch/wait and
// readback, followed by the OpenCL event durations for kernel and readback.
void atlas_trace_profile_last(double* times) noexcept {
    std::copy_n(trace_times, 5, times);
}
#endif

// Inputs come from the bounded symbolic batch prepared by the native tracer.
// Only a successful readback commits output to the host.
int atlas_trace_screen_opencl_index(uint32_t prime, uint32_t pairs, uint32_t width,
        const uint32_t* initial, const uint32_t* offsets, const uint32_t* pivot,
        const uint32_t* columns, const uint32_t* coefficients, uint32_t* remainders, size_t index) noexcept {
    try {
        if (!pairs || !width || prime < 2 || prime >= (1u << 31)) return 1;
        std::lock_guard<std::mutex> guard(lock);
        Runtime& rt = runtime(index);
#ifdef ATLAS_OPENCL_PROFILE
        const auto t0 = Clock::now();
        Event run, read;
        cl_event* run_event = &run.value;
        cl_event* read_event = &read.value;
#else
        cl_event* run_event = nullptr;
        cl_event* read_event = nullptr;
#endif
        const size_t size = static_cast<size_t>(pairs) * width;
        const size_t entries = offsets[width];
        const size_t local_size = ((size_t(width) + 1) / 2 + rt.screen_group) * sizeof(cl_ulong);
        const bool local = rt.screen_group && local_size <= rt.local_bytes;
        Buffer input[7], work, result;
        const uint32_t unused = 0;
        const uint32_t* src[] = {initial, offsets, pivot, entries ? columns : &unused,
                                 entries ? coefficients : &unused, nullptr, nullptr};
        size_t lengths[] = {size, size_t(width) + 1, width, std::max<size_t>(entries, 1),
                            std::max<size_t>(entries, 1), width, 0};
        std::vector<uint32_t> starts, parents, factors, order, levels;
        cl_uint depth = 0;
        if (local) {
            starts.resize(size_t(width) + 1);
            parents.resize(lengths[3]);
            factors.resize(lengths[4]);
            order.resize(width);
            levels.resize(size_t(width) + 1);
            depth = atlas::trace_levels(width, offsets, columns, coefficients, starts.data(),
                                        parents.data(), factors.data(), order.data(), levels.data());
            if (!depth) return 1;
            src[1] = starts.data();
            src[3] = parents.data();
            src[4] = factors.data();
            src[5] = order.data();
            src[6] = levels.data();
            lengths[6] = depth + 1;
        }
        cl_int error;
        cl_kernel k = local ? rt.screen_levels.value : rt.screen.value;
        arg(k, 0, prime);
        arg(k, 1, pairs);
        arg(k, 2, width);
        for (cl_uint i = 0; i < (local ? 7u : 5u); ++i) {
            input[i].value = clCreateBuffer(rt.context.value, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            lengths[i]*sizeof(cl_uint), const_cast<uint32_t*>(src[i]), &error);
            check(error);
            arg(k, i < 5 ? 3+i : 5+i, input[i].value);
        }
        if (local) {
            check(clSetKernelArg(k, 8, local_size, nullptr));
            arg(k, 12, depth);
        } else {
            work.value = clCreateBuffer(rt.context.value, CL_MEM_READ_WRITE, size*sizeof(cl_ulong), nullptr, &error);
            check(error);
            arg(k, 8, work.value);
        }
        result.value = clCreateBuffer(rt.context.value, CL_MEM_WRITE_ONLY, size*sizeof(cl_uint), nullptr, &error);
        check(error);
        arg(k, 9, result.value);
        const size_t global = local ? pairs * rt.screen_group : pairs;
#ifdef ATLAS_OPENCL_PROFILE
        check(clFinish(rt.queue.value));
        const auto t1 = Clock::now();
#endif
        check(clEnqueueNDRangeKernel(rt.queue.value, k, 1, nullptr, &global,
                                     local ? &rt.screen_group : nullptr, 0, nullptr, run_event));
#ifdef ATLAS_OPENCL_PROFILE
        check(clWaitForEvents(1, run_event));
        const auto t2 = Clock::now();
#endif
        check(clEnqueueReadBuffer(rt.queue.value, result.value, CL_TRUE, 0, size*sizeof(cl_uint),
                                  remainders, 0, nullptr, read_event));
#ifdef ATLAS_OPENCL_PROFILE
        const auto t3 = Clock::now();
        trace_times[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        trace_times[1] = std::chrono::duration<double, std::milli>(t2 - t1).count();
        trace_times[2] = std::chrono::duration<double, std::milli>(t3 - t2).count();
        trace_times[3] = event_ms(run.value);
        trace_times[4] = event_ms(read.value);
#endif
        return 0;
    } catch (...) {
        return 1;
    }
}

// Run code[0, length) for `count` primes. `pools` holds count*pool words, the
// generators already in place; `status` receives 1 for every prime whose run
// passed all checks. Returns 0 when the batch was run on the device.
int atlas_replay_opencl_index(const uint32_t* code, size_t length, uint32_t pool, uint32_t work, const uint32_t* primes,
                              uint32_t count, uint32_t* pools, uint32_t* status, size_t index) noexcept {
    try {
        if (!count || length > 0xFFFFFFFFu) return 1;
        std::lock_guard<std::mutex> guard(lock);
        Runtime& rt = runtime(index);
        cl_int error;
        const size_t pool_bytes = static_cast<size_t>(count) * std::max<uint32_t>(pool, 1) * sizeof(uint32_t);
        const size_t work_bytes = static_cast<size_t>(count) * std::max<uint32_t>(work, 1) * sizeof(cl_ulong);
        std::vector<uint32_t> ok(count, 1);
        Buffer program, moduli, coefficients, scratch, flags;
        program.value = clCreateBuffer(rt.context.value, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       std::max<size_t>(length, 1) * sizeof(uint32_t), const_cast<uint32_t*>(code), &error);
        check(error);
        moduli.value = clCreateBuffer(rt.context.value, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      count * sizeof(uint32_t), const_cast<uint32_t*>(primes), &error);
        check(error);
        coefficients.value = clCreateBuffer(rt.context.value, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pool_bytes, pools, &error);
        check(error);
        scratch.value = clCreateBuffer(rt.context.value, CL_MEM_READ_WRITE, work_bytes, nullptr, &error);
        check(error);
        flags.value = clCreateBuffer(rt.context.value, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     count * sizeof(uint32_t), ok.data(), &error);
        check(error);
        cl_kernel kernel = rt.kernel.value;
        arg(kernel, 0, program.value);
        arg<cl_uint>(kernel, 3, pool);
        arg<cl_uint>(kernel, 4, work);
        arg(kernel, 5, moduli.value);
        arg(kernel, 6, coefficients.value);
        arg(kernel, 7, scratch.value);
        arg(kernel, 8, flags.value);
        // Whole blocks at a time, so that no single launch runs for long.
        const size_t global = count;
        const size_t slice = 1u << 20;
        size_t from = 0;
        while (from < length) {
            size_t to = from, at = from;
            while (at < length) {
                const uint32_t op = code[at];
                size_t next;
                if (op == 1) {
                    if (at > from && at - from >= slice) break;  // a block starts here
                    next = at + 2;
                } else if (op == 2 || op == 3 || op == 6) next = at + 3 + code[at + 2];
                else if (op == 4) next = at + 4 + code[at + 3];
                else if (op == 5) next = at + 2 + code[at + 1];
                else return 1;
                at = to = next;
            }
            arg<cl_uint>(kernel, 1, static_cast<cl_uint>(from));
            arg<cl_uint>(kernel, 2, static_cast<cl_uint>(to));
            check(clEnqueueNDRangeKernel(rt.queue.value, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr));
            from = to;
        }
        check(clEnqueueReadBuffer(rt.queue.value, flags.value, CL_TRUE, 0, count * sizeof(uint32_t), ok.data(), 0, nullptr, nullptr));
        check(clEnqueueReadBuffer(rt.queue.value, coefficients.value, CL_TRUE, 0, pool_bytes, pools, 0, nullptr, nullptr));
        std::copy(ok.begin(), ok.end(), status);
        return 0;
    } catch (...) {
        return 1;
    }
}

int atlas_replay_opencl(const uint32_t* code, size_t length, uint32_t pool, uint32_t work, const uint32_t* primes,
                        uint32_t count, uint32_t* pools, uint32_t* status) noexcept {
    return atlas_replay_opencl_index(code, length, pool, work, primes, count, pools, status, default_device());
}

// 1 when an OpenCL GPU is usable for replays.
int atlas_replay_opencl_available() noexcept {
    try {
        std::lock_guard<std::mutex> guard(lock);
        runtime(default_device());
        return 1;
    } catch (...) {
        return 0;
    }
}

}
