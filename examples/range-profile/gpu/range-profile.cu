#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>

#define CUDA_CHECK(X) \
    do { \
        cudaError_t err = (X); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUFFT_CHECK(X) \
    do { \
        cufftResult err = (X); \
        if (err != CUFFT_SUCCESS) { \
            std::cerr << "CUFFT error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << err << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


class GpuTimer
{
    private:
        cudaEvent_t start_event;
        cudaEvent_t stop_event;
    public:
        GpuTimer()
        {
            CUDA_CHECK(cudaEventCreate(&start_event));
            CUDA_CHECK(cudaEventCreate(&stop_event));
        }

        ~GpuTimer()
        {
            CUDA_CHECK(cudaEventDestroy(start_event));
            CUDA_CHECK(cudaEventDestroy(stop_event));
        }

        void start()
        {
            CUDA_CHECK(cudaEventRecord(start_event, 0));
        }

        float stop()
        {
            CUDA_CHECK(cudaEventRecord(stop_event, 0));
            CUDA_CHECK(cudaEventSynchronize(stop_event));

            float elapsed;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));
            return elapsed;
        }
};

constexpr int NUM_PULSES = 1024;
constexpr int NUM_SAMPLES = 8192;

constexpr float RANGE_FREQ = 0.08f;
constexpr float DOPPLER_FREQ = 0.12f;

constexpr int TOTAL_SAMPLES = NUM_PULSES * NUM_SAMPLES;

using complex_t = cufftComplex;

__global__ void generate_iq_data(cuFloatComplex* output, int pulses, int samples_per_pulse, float range_freq, float doppler_freq)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pulses * samples_per_pulse;

    if (thread_id >= total)
    {
        // Early return as the thread is out of bounds
        return;
    }

    int pulse_idx = thread_id / samples_per_pulse;
    int sample_idx = thread_id % samples_per_pulse;

    float phase = 2.0f * M_PI * (range_freq * sample_idx + doppler_freq * pulse_idx);
    float sinValue, cosValue;
    __sincosf(phase, &sinValue, &cosValue);
    output[thread_id] = make_cuFloatComplex(sinValue, cosValue);
}

__global__ void transpose(complex_t* input, complex_t* output, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    output[x * rows + y] =
        input[y * cols + x];
}

__global__ void compute_magnitude(complex_t* input, float* output, int total_samples)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= total_samples)
    {
        // Early return as the thread is out of bounds
        return;
    }

    cuFloatComplex val = input[thread_id];
    output[thread_id] = sqrtf(val.x * val.x + val.y * val.y);
}

void export_csv(
    const std::vector<float>& data,
    int rows,
    int cols,
    const std::string& filename)
{
    std::ofstream file(filename);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {

            file << data[r * cols + c];

            if (c != cols - 1)
                file << ",";
        }

        file << "\n";
    }

    std::cout << "Exported: " << filename << std::endl;
}

std::string generate_timestamp()
{
    auto now =
        std::chrono::system_clock::now();

    auto time =
        std::chrono::system_clock::to_time_t(now);

    std::tm local_time = *std::localtime(&time);

    std::stringstream ss;

    ss << std::put_time(
        &local_time,
        "%Y-%m-%d_%H-%M-%S");

    return ss.str();
}

int main()
{
    std::cout << "Range-Doppler CUDA MVP\n";

    ////////////////////////////////////////////////////////////////////////////
    // Device Buffers
    ////////////////////////////////////////////////////////////////////////////
    GpuTimer mainTimer;
    mainTimer.start();
    complex_t* d_iq = nullptr;
    complex_t* d_transposed = nullptr;
    float* d_magnitude = nullptr;

    CUDA_CHECK(cudaMalloc(&d_iq,
                          sizeof(complex_t) * TOTAL_SAMPLES));

    CUDA_CHECK(cudaMalloc(&d_transposed,
                          sizeof(complex_t) * TOTAL_SAMPLES));

    CUDA_CHECK(cudaMalloc(&d_magnitude,
                          sizeof(float) * TOTAL_SAMPLES));

    ////////////////////////////////////////////////////////////////////////////
    // Generate Synthetic IQ
    ////////////////////////////////////////////////////////////////////////////

    {
        constexpr int THREADS = 256;

        int blocks =
            (TOTAL_SAMPLES + THREADS - 1) / THREADS;

        std::cout << "Generating synthetic IQ data for " << NUM_PULSES << " pulses and "
                  << NUM_SAMPLES << " samples per pulse...\n";
        std::cout << "Total Complex Samples: " << TOTAL_SAMPLES << "\n";

        GpuTimer timer;
        timer.start();
        generate_iq_data<<<blocks, THREADS>>>(
            d_iq,
            NUM_PULSES,
            NUM_SAMPLES,
            RANGE_FREQ,
            DOPPLER_FREQ);

        CUDA_CHECK(cudaDeviceSynchronize());
        float elapsed = timer.stop();
        std::cout << "Generated synthetic IQ in " << elapsed << " ms\n";
    }

    ////////////////////////////////////////////////////////////////////////////
    // Range FFT
    ////////////////////////////////////////////////////////////////////////////

    cufftHandle range_plan;

    {
        int rank = 1;
        int n[] = { NUM_SAMPLES };

        int istride = 1;
        int ostride = 1;

        int idist = NUM_SAMPLES;
        int odist = NUM_SAMPLES;

        int inembed[] = { NUM_SAMPLES };
        int onembed[] = { NUM_SAMPLES };

        int batch = NUM_PULSES;


        // Planning is a very expensive operation, so we time it separately from execution
        CUFFT_CHECK(
            cufftPlanMany(
                &range_plan,
                rank,
                n,
                inembed,
                istride,
                idist,
                onembed,
                ostride,
                odist,
                CUFFT_C2C,
                batch));

                
        GpuTimer timer;
        timer.start();
        CUFFT_CHECK(
            cufftExecC2C(
                range_plan,
                d_iq,
                d_iq,
                CUFFT_FORWARD));

        CUDA_CHECK(cudaDeviceSynchronize());
        float elapsed = timer.stop();
        std::cout << "Completed range FFT in " << elapsed << " ms\n";
    }

    ////////////////////////////////////////////////////////////////////////////
    // Transpose
    ////////////////////////////////////////////////////////////////////////////

    {
        dim3 threads(16, 16);

        dim3 blocks(
            (NUM_SAMPLES + threads.x - 1) / threads.x,
            (NUM_PULSES + threads.y - 1) / threads.y);

            GpuTimer timer;
            timer.start();
        transpose<<<blocks, threads>>>(
            d_iq,
            d_transposed,
            NUM_PULSES,
            NUM_SAMPLES);

        CUDA_CHECK(cudaDeviceSynchronize());
        float elapsed = timer.stop();
        std::cout << "Completed transpose in " << elapsed << " ms\n";
    }

    ////////////////////////////////////////////////////////////////////////////
    // Doppler FFT
    ////////////////////////////////////////////////////////////////////////////

    cufftHandle doppler_plan;

    {
        int rank = 1;
        int n[] = { NUM_PULSES };

        int istride = 1;
        int ostride = 1;

        int idist = NUM_PULSES;
        int odist = NUM_PULSES;

        int inembed[] = { NUM_PULSES };
        int onembed[] = { NUM_PULSES };

        int batch = NUM_SAMPLES;
        GpuTimer timer;
        timer.start();
        CUFFT_CHECK(
            cufftPlanMany(
                &doppler_plan,
                rank,
                n,
                inembed,
                istride,
                idist,
                onembed,
                ostride,
                odist,
                CUFFT_C2C,
                batch));

        CUFFT_CHECK(
            cufftExecC2C(
                doppler_plan,
                d_transposed,
                d_transposed,
                CUFFT_FORWARD));

        CUDA_CHECK(cudaDeviceSynchronize());

        float elapsed = timer.stop();
        std::cout << "Completed doppler FFT in " << elapsed << " ms\n";
    }

    ////////////////////////////////////////////////////////////////////////////
    // Magnitude
    ////////////////////////////////////////////////////////////////////////////

    {
        constexpr int THREADS = 256;

        int blocks =
            (TOTAL_SAMPLES + THREADS - 1) / THREADS;

            GpuTimer timer;
            timer.start();
        compute_magnitude<<<blocks, THREADS>>>(
            d_transposed,
            d_magnitude,
            TOTAL_SAMPLES);

        CUDA_CHECK(cudaDeviceSynchronize());

        float elapsed = timer.stop();
        std::cout << "Computed magnitude in " << elapsed << " ms\n";
    }

    ////////////////////////////////////////////////////////////////////////////
    // Copy Back
    ////////////////////////////////////////////////////////////////////////////

    std::vector<float> host_magnitude(TOTAL_SAMPLES);

    CUDA_CHECK(
        cudaMemcpy(
            host_magnitude.data(),
            d_magnitude,
            sizeof(float) * TOTAL_SAMPLES,
            cudaMemcpyDeviceToHost));

    ////////////////////////////////////////////////////////////////////////////
    // Cleanup
    ////////////////////////////////////////////////////////////////////////////

    cufftDestroy(range_plan);
    cufftDestroy(doppler_plan);

    cudaFree(d_iq);
    cudaFree(d_transposed);
    cudaFree(d_magnitude);
    float mainTime = mainTimer.stop();
    std::cout << "Done in " << mainTime << "ms\n";

        ////////////////////////////////////////////////////////////////////////////
    // Export CSV
    ////////////////////////////////////////////////////////////////////////////

    namespace fs = std::filesystem;

    // Assumes executable is launched from build/
    // and source tree lives one level above
    fs::path output_dir =
        fs::current_path().parent_path() / "data";

    fs::create_directories(output_dir);

    std::string filename =
        "range_doppler_" +
        generate_timestamp() +
        ".csv";

    fs::path output_path =
        output_dir / filename;

    export_csv(
        host_magnitude,
        NUM_SAMPLES,
        NUM_PULSES,
        output_path.string());

    return 0;
}