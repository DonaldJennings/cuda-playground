#include <cuda_runtime.h>
#include <cufft.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// ── Error checking ─────────────────────────────────────────────────────────────

#define CUDA_CHECK(X) \
    do { \
        cudaError_t _e = (X); \
        if (_e != cudaSuccess) { \
            std::cerr << "CUDA error " << __FILE__ << ':' << __LINE__ \
                      << " – " << cudaGetErrorString(_e) << '\n'; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUFFT_CHECK(X) \
    do { \
        cufftResult _e = (X); \
        if (_e != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT error " << __FILE__ << ':' << __LINE__ \
                      << " – code " << _e << '\n'; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ── Problem dimensions ─────────────────────────────────────────────────────────

using complex_t = cufftComplex;

// Tile size for the shared-memory transpose kernel.
// 32 = warp width, giving fully coalesced global reads and writes.
constexpr int TILE     = 32;
constexpr int TILE_PAD = 1;   // eliminates shared-memory bank conflicts (see below)

// ── RAII GPU event timer ───────────────────────────────────────────────────────

class GpuTimer
{
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
public:
    GpuTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }

    // Records stop event, synchronises CPU to it, returns elapsed ms.
    float elapsed(cudaStream_t s = 0)
    {
        CUDA_CHECK(cudaEventRecord(stop_, s));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ── Kernels ────────────────────────────────────────────────────────────────────

// IQ generation with a 2D grid: blockIdx.x covers samples (fast time),
// blockIdx.y covers pulses (slow time). This eliminates the expensive integer
// division and modulo that a 1D flat layout requires at 8192 samples/pulse.
__global__ void k_generate_iq(
    complex_t* __restrict__ out,
    int   samples_per_pulse,
    float range_freq,
    float doppler_freq)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;  // sample index
    const int p = blockIdx.y;                               // pulse  index

    if (s >= samples_per_pulse) return;

    // e^(j·phase) = cos(phase) + j·sin(phase)
    constexpr float TWO_PI = 2.f * 3.14159265358979323846f;
    const float phase = TWO_PI * (range_freq * s + doppler_freq * p);
    float sin_v, cos_v;
    __sincosf(phase, &sin_v, &cos_v);

    out[p * samples_per_pulse + s] = make_cuFloatComplex(cos_v, sin_v);
}

// Amplitude of every complex element in the Doppler-FFT output.
__global__ void k_magnitude(
    const complex_t* __restrict__ in,
    float* __restrict__           out,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const complex_t v = in[i];
    out[i] = sqrtf(v.x * v.x + v.y * v.y);
}

// --- Radar Configuration struct with default parameters. Can be overridden via command-line args. ---

struct RadarConfig
{
    int pulses = 1024;
    int samples = 8192;
    float range_freq = 0.08f;
    float doppler_freq = 0.12f;

    int iterations = 100;
};

// ── Per-stage timing breakdown ─────────────────────────────────────────────────

struct PipelineTiming
{
    float iq_gen_ms      = 0.f;
    float range_fft_ms   = 0.f;
    float doppler_fft_ms = 0.f;
    float magnitude_ms   = 0.f;
    float total_ms       = 0.f;

    void print() const
    {
        const auto pct = [&](float t)
        {
            return total_ms > 0.f ? 100.f * t / total_ms : 0.f;
        };

        std::cout << std::fixed << std::setprecision(3);
        std::cout
            << "  IQ generation : " << std::setw(8) << iq_gen_ms      << " ms  (" << pct(iq_gen_ms)      << " %)\n"
            << "  Range FFT     : " << std::setw(8) << range_fft_ms   << " ms  (" << pct(range_fft_ms)   << " %)\n"
            << "  Doppler FFT   : " << std::setw(8) << doppler_fft_ms << " ms  (" << pct(doppler_fft_ms) << " %)\n"
            << "  Magnitude     : " << std::setw(8) << magnitude_ms   << " ms  (" << pct(magnitude_ms)   << " %)\n"
            << "  ──────────────────────────────────────────────\n"
            << "  Total         : " << std::setw(8) << total_ms       << " ms\n";
    }
};

// ── Range-Doppler processor ────────────────────────────────────────────────────

class RangeDopplerProcessor
{
public:
    RangeDopplerProcessor(RadarConfig config = {}) : config_(config)
    {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        allocate_buffers();
        create_plans();
    }

    ~RangeDopplerProcessor()
    {
        cufftDestroy(range_plan_);
        cufftDestroy(doppler_plan_);
        cudaFree(d_iq_);
        cudaFree(d_magnitude_);
        cudaFreeHost(h_magnitude_);
        cudaStreamDestroy(stream_);
    }

    // Run one pipeline pass without timing to prime cuFFT internal state,
    // JIT compilation, and the GPU cache hierarchy.
    void warmup()
    {
        run_pipeline();
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // Single pass with per-stage GPU event timings.
    // Note: each elapsed() call inserts a host sync, which adds pipeline
    // bubbles; the total here will be higher than benchmark() reports.
    PipelineTiming profile()
    {
        PipelineTiming t;
        GpuTimer timer;

        timer.start(stream_);
        launch_iq_gen();
        t.iq_gen_ms = timer.elapsed(stream_);

        timer.start(stream_);
        CUFFT_CHECK(cufftExecC2C(range_plan_, d_iq_, d_iq_, CUFFT_FORWARD));
        t.range_fft_ms = timer.elapsed(stream_);

        timer.start(stream_);
        CUFFT_CHECK(cufftExecC2C(doppler_plan_, d_iq_, d_iq_, CUFFT_FORWARD));
        t.doppler_fft_ms = timer.elapsed(stream_);

        timer.start(stream_);
        launch_magnitude();
        t.magnitude_ms = timer.elapsed(stream_);

        t.total_ms = t.iq_gen_ms + t.range_fft_ms + t.doppler_fft_ms + t.magnitude_ms;

        // Copy result to pinned host buffer via async DMA.
        CUDA_CHECK(cudaMemcpyAsync(
            h_magnitude_, d_magnitude_,
            sizeof(float) * (config_.pulses * config_.samples),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        return t;
    }

    // Run the pipeline N times back-to-back with a single event pair wrapping
    // the entire batch. No intra-pipeline synchronisation → GPU runs at full
    // throughput. Reports average latency and effective memory bandwidth.
    void benchmark(int iterations = 100)
    {
        GpuTimer timer;
        timer.start(stream_);

        for (int i = 0; i < iterations; ++i)
            run_pipeline();

        const float total_ms = timer.elapsed(stream_);
        const float avg_ms   = total_ms / float(iterations);

        // Rough bandwidth estimate: account for the dominant memory traffic –
        // two in-place FFT passes (~2 R+W each over 64 MB) + transpose (1R+1W)
        // + magnitude (1R + 0.5W) ≈ 5.5 passes × 64 MB = 352 MB per iteration.
        const double bytes = 5.5 * (config_.pulses * config_.samples) * sizeof(complex_t);
        const double bw    = bytes / (avg_ms * 1e-3) / 1e9;

        std::cout << std::fixed << std::setprecision(3)
                  << "Benchmark (" << iterations << " iterations):\n"
                  << "  Total   : " << total_ms << " ms\n"
                  << "  Average : " << avg_ms   << " ms / iter\n"
                  << "  Eff. BW : " << bw       << " GB/s\n";
    }

    const float* host_magnitude() const { return h_magnitude_; }

private:
    // ── Kernel launch helpers ────────────────────────────────────────────────

    void launch_iq_gen()
    {
        // One thread per (sample, pulse) pair. X-dimension covers samples so
        // writes within a warp are contiguous → coalesced.
        const dim3 threads(256, 1);
        const dim3 blocks((config_.samples + 255) / 256, config_.pulses);
        k_generate_iq<<<blocks, threads, 0, stream_>>>(
            d_iq_, config_.samples, config_.range_freq, config_.doppler_freq);
    }

    void launch_magnitude()
    {
        const dim3 blocks((config_.pulses * config_.samples + 255) / 256);
        k_magnitude<<<blocks, 256, 0, stream_>>>(
            d_iq_, d_magnitude_, config_.pulses * config_.samples);
    }

    void run_pipeline()
    {
        launch_iq_gen();
        CUFFT_CHECK(cufftExecC2C(range_plan_, d_iq_, d_iq_, CUFFT_FORWARD));
        CUFFT_CHECK(cufftExecC2C(doppler_plan_, d_iq_, d_iq_, CUFFT_FORWARD));
        launch_magnitude();
    }

    // ── Setup ────────────────────────────────────────────────────────────────

    void allocate_buffers()
    {
        CUDA_CHECK(cudaMalloc(&d_iq_,         sizeof(complex_t) * (config_.pulses * config_.samples)));
        CUDA_CHECK(cudaMalloc(&d_magnitude_,  sizeof(float)     * (config_.pulses * config_.samples)));

        // Pinned (page-locked) host buffer enables direct DMA for the DtH copy,
        // avoiding the driver's internal staging bounce through a pinned region.
        CUDA_CHECK(cudaMallocHost(&h_magnitude_, sizeof(float) * (config_.pulses * config_.samples)));
    }

    void create_plans()
    {
        // Range FFT: NUM_PULSES batches of FFT-NUM_SAMPLES.
        // Data layout: d_iq_[p * NUM_SAMPLES + s] (pulse-major).
        {
            int n[]   = { config_.samples };
            int emb[] = { config_.samples };
            CUFFT_CHECK(cufftPlanMany(
                &range_plan_, 1, n,
                emb, /*istride=*/1, /*idist=*/config_.samples,
                emb, /*ostride=*/1, /*odist=*/config_.samples,
                CUFFT_C2C, /*batch=*/config_.pulses));
            CUFFT_CHECK(cufftSetStream(range_plan_, stream_));
        }

        // Doppler FFT: NUM_SAMPLES batches of FFT-NUM_PULSES.
        // Data layout after transpose: d_transposed_[s * NUM_PULSES + p] (sample-major).
        {
            int n[]   = { config_.pulses };
            int emb[] = { config_.pulses };
            CUFFT_CHECK(cufftPlanMany(
                &doppler_plan_, 1, n,
                emb, /*istride=*/config_.samples, /*idist=*/1,
                emb, /*ostride=*/config_.samples, /*odist=*/1,
                CUFFT_C2C, /*batch=*/config_.samples));
            CUFFT_CHECK(cufftSetStream(doppler_plan_, stream_));
        }
    }

    RadarConfig config_;

    cudaStream_t stream_       = 0;
    complex_t*   d_iq_         = nullptr;
    float*       d_magnitude_  = nullptr;
    float*       h_magnitude_  = nullptr;
    cufftHandle  range_plan_   = 0;
    cufftHandle  doppler_plan_ = 0;
};

// ── I/O helpers ────────────────────────────────────────────────────────────────

static void export_csv(
    const float*       data,
    int                rows,
    int                cols,
    const std::string& path)
{
    std::ofstream f(path);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            f << data[r * cols + c];
            if (c + 1 < cols) f << ',';
        }
        f << '\n';
    }
    std::cout << "Exported: " << path << '\n';
}

static std::string timestamp()
{
    const auto t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm tm = *std::localtime(&t);
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}


// ── main ───────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    RadarConfig config;
    if (argc > 1)
    {
        config.pulses = std::stoi(argv[1]);
    }
    if (argc > 2)
    {
        config.samples = std::stoi(argv[2]);
    }
    if (argc > 3)
    {
        config.range_freq = std::stof(argv[3]);
    }
    if (argc > 4)
    {
        config.doppler_freq = std::stof(argv[4]);
    }
    if (argc > 5)
    {
        config.iterations = std::stoi(argv[5]);
    }

    std::cout << "Radar Configuration:\n"
              << "  Pulses       : " << config.pulses << '\n'
              << "  Samples/Pulse: " << config.samples << '\n'
              << "  Range Freq   : " << config.range_freq << '\n'
              << "  Doppler Freq : " << config.doppler_freq << '\n'
              << "  Iterations   : " << config.iterations << "\n\n";

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Device : " << prop.name << '\n'
              << "SMs    : " << prop.multiProcessorCount << '\n'
              << "SM arch: " << prop.major << '.' << prop.minor << "\n\n";

    RangeDopplerProcessor proc(config);

    std::cout << "Warming up...\n";
    proc.warmup();

    std::cout << "\nPer-stage breakdown (single pass, includes sync overhead):\n";
    const auto t = proc.profile();
    t.print();

    std::cout << '\n';
    proc.benchmark(100);
    std::cout << '\n';

    namespace fs = std::filesystem;
    fs::path out_dir = fs::current_path() / "data";
    fs::create_directories(out_dir);
    const fs::path out_path =
        out_dir / ("range_doppler_" + timestamp() + ".csv");

    export_csv(proc.host_magnitude(), config.pulses, config.samples, out_path.string());

    return 0;
}
