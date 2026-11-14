// Optional OpenCL 1.2 plugin. Input and output are ordinary host buffers.
// The portable CPU library has no link-time dependency on an OpenCL runtime.
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const char* source = R"CLC(
// Both primes are part of the host ABI. Constant divisors let GPU compilers
// lower the 64-bit remainder to multiplication by a reciprocal.
uint residue(ulong a, uint prime) {
    return prime==1000000007u ? (uint)(a%1000000007ul) : (uint)(a%1000000009ul);
}
kernel void pivot(global uint *a, global int *pivots, int rows, int cols, int col, uint prime) {
    local int selected;
    local uint scale;
    if (get_local_id(0)==0) {
        int row=col;
        if (col && pivots[col-1]<0) row=rows;
        while (row<rows && !a[row*cols+col]) ++row;
        selected = row==rows ? -1 : row;
        pivots[col] = selected;
        if (row<rows) {
            uint base=a[row*cols+col], inv=1;
            for (uint exp=prime-2; exp; exp>>=1) {
                if (exp & 1) inv=residue((ulong)inv*base,prime);
                base=residue((ulong)base*base,prime);
            }
            scale=inv;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (selected<0) return;
    for (int j=col+(int)get_local_id(0); j<cols; j+=(int)get_local_size(0)) {
        uint old=a[selected*cols+j];
        if (selected!=col) a[selected*cols+j]=a[col*cols+j];
        a[col*cols+j]=residue((ulong)old*scale,prime);
    }
}
kernel void eliminate(global uint *a, global const int *pivots,
                      int rows, int cols, int col, uint prime) {
    int j=col+1+(int)get_global_id(0), i=col+1+(int)get_global_id(1);
    if (pivots[col]<0 || i>=rows || j>=cols) return;
    uint factor=a[i*cols+col];
    if (!factor) return;
    uint delta=residue((ulong)factor*a[col*cols+j],prime);
    uint entry=a[i*cols+j];
    a[i*cols+j]=entry>=delta ? entry-delta : entry+prime-delta;
}
// One work-item computes one strict-transform exponent vector. This uses
// signed 32-bit arithmetic with an a priori checked host bound.
kernel void chart_terms(global const int *exponents, global const int *rays,
                        global const int *cones, global const int *orders,
                        global int *output, int terms, int dimension) {
    size_t pair=get_global_id(0);
    int term=pair%terms, chart=pair/terms;
    if (dimension==3) {
        int3 e=vload3(term,exponents);
        for (int axis=0;axis<3;++axis) {
            int index=cones[3*chart+axis];
            int3 r=vload3(index,rays);
            output[3*pair+axis]=e.x*r.x+e.y*r.y+e.z*r.z-orders[index];
        }
        return;
    }
    if (dimension==4) {
        int4 e=vload4(term,exponents);
        for (int axis=0;axis<4;++axis) {
            int index=cones[4*chart+axis];
            int4 r=vload4(index,rays);
            output[4*pair+axis]=e.x*r.x+e.y*r.y+e.z*r.z+e.w*r.w-orders[index];
        }
        return;
    }
    for (int axis=0;axis<dimension;++axis) {
        int index=cones[dimension*chart+axis], value=0;
        for(int j=0;j<dimension;++j)
            value+=exponents[dimension*term+j]*rays[dimension*index+j];
        output[dimension*pair+axis]=value-orders[index];
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
using Kernel = Owned<cl_kernel, clReleaseKernel>;
using Queue = Owned<cl_command_queue, clReleaseCommandQueue>;

const std::vector<cl_device_id>& devices() {
    static const std::vector<cl_device_id> result = [] {
        std::vector<cl_device_id> found;
        cl_uint count = 0;
        if (clGetPlatformIDs(0,nullptr,&count)!=CL_SUCCESS || !count) return found;
        std::vector<cl_platform_id> platforms(count);
        check(clGetPlatformIDs(count,platforms.data(),nullptr));
        for (auto platform : platforms) {
            cl_uint n = 0;
            if (clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,nullptr,&n)!=CL_SUCCESS || !n) continue;
            std::vector<cl_device_id> ids(n);
            check(clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,n,ids.data(),nullptr));
            found.insert(found.end(),ids.begin(),ids.end());
        }
        return found;
    }();
    return result;
}

std::string info(cl_device_id device, cl_device_info key) {
    size_t size = 0;
    check(clGetDeviceInfo(device,key,0,nullptr,&size));
    std::vector<char> data(size);
    check(clGetDeviceInfo(device,key,size,data.data(),nullptr));
    return std::string(data.data());
}

struct Runtime {
    Owned<cl_context, clReleaseContext> context;
    Owned<cl_program, clReleaseProgram> program;
    cl_device_id device;
    explicit Runtime(cl_device_id id) : device(id) {
        cl_int status;
        context.value = clCreateContext(nullptr,1,&id,nullptr,nullptr,&status);
        check(status);
        program.value = clCreateProgramWithSource(context.value,1,&source,nullptr,&status);
        check(status);
        check(clBuildProgram(program.value,1,&id,"-cl-std=CL1.2",nullptr,nullptr));
    }
};

Runtime& runtime(int device) {
    // Programs are immutable after building. Each request gets its own queue,
    // kernels and buffers so setting kernel arguments is thread-safe.
    static std::mutex lock;
    static std::map<int, std::unique_ptr<Runtime>> instances;
    std::lock_guard<std::mutex> guard(lock);
    auto& instance = instances[device];
    if (!instance) instance = std::make_unique<Runtime>(devices().at(device));
    return *instance;
}

template<typename T> void arg(cl_kernel kernel, cl_uint index, T value) {
    check(clSetKernelArg(kernel,index,sizeof(T),&value));
}

int gpu_rank(const uint32_t* matrix, int rows, int cols, uint32_t prime, int device, int* minor) {
    if (!matrix || !minor || rows<cols || cols<1 || cols>512 || rows>16384 ||
        (prime!=1000000007 && prime!=1000000009)) return -1;
    auto& rt = runtime(device);
    cl_int status;
    Queue queue;
    queue.value=clCreateCommandQueue(rt.context.value,rt.device,0,&status);
    check(status);
    Buffer a, p;
    a.value=clCreateBuffer(rt.context.value,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                           size_t(rows)*cols*sizeof(cl_uint),const_cast<uint32_t*>(matrix),&status);
    check(status);
    p.value=clCreateBuffer(rt.context.value,CL_MEM_READ_WRITE,cols*sizeof(cl_int),nullptr,&status);
    check(status);
    Kernel pivot, eliminate;
    pivot.value=clCreateKernel(rt.program.value,"pivot",&status); check(status);
    eliminate.value=clCreateKernel(rt.program.value,"eliminate",&status); check(status);
    for (auto k : {pivot.value,eliminate.value}) {
        arg(k,0,a.value); arg(k,1,p.value);
    }
    for (auto k : {pivot.value,eliminate.value}) {
        arg(k,2,rows); arg(k,3,cols); arg(k,5,prime);
    }
    size_t allowed = 1;
    check(clGetKernelWorkGroupInfo(pivot.value,rt.device,CL_KERNEL_WORK_GROUP_SIZE,
                                  sizeof(allowed),&allowed,nullptr));
    const size_t group = std::min<size_t>(64,allowed);
    for (int col=0; col<cols; ++col) {
        arg(pivot.value,4,col);
        check(clEnqueueNDRangeKernel(queue.value,pivot.value,1,nullptr,&group,&group,0,nullptr,nullptr));
        if (col+1 < cols) {
            size_t grid[]={(size_t(cols-col-1)+31)/32*32, (size_t(rows-col-1)+3)/4*4};
            arg(eliminate.value,4,col);
            check(clEnqueueNDRangeKernel(queue.value,eliminate.value,2,nullptr,grid,nullptr,0,nullptr,nullptr));
        }
    }
    std::vector<cl_int> pivots(cols);
    check(clEnqueueReadBuffer(queue.value,p.value,CL_TRUE,0,cols*sizeof(cl_int),pivots.data(),0,nullptr,nullptr));
    std::vector<int> permutation(rows);
    std::iota(permutation.begin(),permutation.end(),0);
    for (int i=0; i<cols; ++i) {
        if (pivots[i]==-1) return 0;
        if (pivots[i]<i || pivots[i]>=rows) return -1;
        std::swap(permutation[pivots[i]],permutation[i]);
        minor[i] = permutation[i];
    }
    return 1;
}
int gpu_charts(const int32_t* exps,const int32_t* rays,const int32_t* cones,const int32_t* orders,
               int terms,int nrays,int charts,int dimension,int device,int32_t* out) {
    if (!exps || !rays || !cones || !orders || !out || terms<1 || charts<1 || nrays<1 ||
        dimension<2 || dimension>8 || int64_t(terms)*charts>2000000 ||
        int64_t(terms)*charts*dimension>6000000 || nrays>100000) return -1;
    auto& rt = runtime(device);
    cl_int status;
    Queue queue;
    queue.value=clCreateCommandQueue(rt.context.value,rt.device,0,&status); check(status);
    Kernel transform;
    transform.value=clCreateKernel(rt.program.value,"chart_terms",&status); check(status);
    size_t pairs=size_t(terms)*charts;
    const int32_t* inputs[] = {exps,rays,cones,orders};
    size_t counts[] = {size_t(dimension)*terms,size_t(dimension)*nrays,size_t(dimension)*charts,size_t(nrays)};
    std::array<Buffer,5> buffers;
    for (int i=0; i<4; ++i) {
        buffers[i].value=clCreateBuffer(rt.context.value,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                counts[i]*sizeof(cl_int),const_cast<int32_t*>(inputs[i]),&status);
        check(status);
    }
    buffers[4].value=clCreateBuffer(rt.context.value,CL_MEM_WRITE_ONLY,dimension*pairs*sizeof(cl_int),nullptr,&status);
    check(status);
    for (int i=0; i<5; ++i) arg(transform.value,i,buffers[i].value);
    arg(transform.value,5,terms);
    arg(transform.value,6,dimension);
    check(clEnqueueNDRangeKernel(queue.value,transform.value,1,nullptr,&pairs,nullptr,0,nullptr,nullptr));
    check(clEnqueueReadBuffer(queue.value,buffers[4].value,CL_TRUE,0,dimension*pairs*sizeof(cl_int),out,0,nullptr,nullptr));
    return 0;
}
} // namespace

extern "C" int resolution_gpu_count() noexcept {
    try { return static_cast<int>(devices().size()); } catch (...) { return 0; }
}
extern "C" int resolution_gpu_info(int index, char* text, size_t capacity) noexcept {
    try {
        auto id=devices().at(index);
        auto details=info(id,CL_DEVICE_NAME)+"\n"+info(id,CL_DEVICE_VENDOR)+"\n"+
                     info(id,CL_DEVICE_VERSION)+"\n"+info(id,CL_DRIVER_VERSION);
        if (!text || capacity<=details.size()) return -1;
        std::memcpy(text,details.c_str(),details.size()+1);
        return 0;
    } catch (...) { return -1; }
}
extern "C" int resolution_gpu_rank(const uint32_t* matrix,int rows,int cols,uint32_t prime,
                                   int device,int* minor) noexcept {
    try { return gpu_rank(matrix,rows,cols,prime,device,minor); } catch (...) { return -1; }
}
extern "C" int resolution_gpu_charts(const int32_t* exps,const int32_t* rays,const int32_t* cones,
                                     const int32_t* orders,int terms,int nrays,int charts,
                                     int device,int32_t* out) noexcept {
    try { return gpu_charts(exps,rays,cones,orders,terms,nrays,charts,3,device,out); }
    catch (...) { return -1; }
}

extern "C" int resolution_gpu_charts_nd(const int32_t* exps,const int32_t* rays,const int32_t* cones,
                                        const int32_t* orders,int terms,int nrays,int charts,
                                        int dimension,int device,int32_t* out) noexcept {
    try { return gpu_charts(exps,rays,cones,orders,terms,nrays,charts,dimension,device,out); }
    catch (...) { return -1; }
}
