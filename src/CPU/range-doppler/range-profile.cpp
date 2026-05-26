#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <fstream>

#include <fftw3.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief RangeDopplerProcessor class encapsulates the entire process of generating synthetic IQ data,
 * performing range and Doppler FFTs, and computing the magnitude of the range-Doppler map.
 */
class RangeDopplerProcessor
{
private:
    constexpr static float PI = 3.14159265358979323846f;

public:
    /**
     * @brief Constructor for the RangeDopplerProcessor class. This constructor
     * will preallocate memory for the input IQ data, intermediate FFT outputs, and the final magnitude array.
     * It also initializes the FFTW plans for both the range and Doppler FFTs.
     * @param pulses The number of radar pulses (slow time dimension)
     * @param samplesPerPulse The number of samples per pulse (fast time dimension)
     */
    RangeDopplerProcessor(
        int pulses,
        int samplesPerPulse)
        : pulses(pulses),
          samplesPerPulse(samplesPerPulse),
          totalSamples(pulses * samplesPerPulse)
    {
        std::cout
            << "Allocating "
            << totalSamples
            << " complex samples\n";

        iq_data = reinterpret_cast<fftwf_complex*>(
            fftwf_malloc(sizeof(fftwf_complex) * totalSamples));

        range_fft_output = reinterpret_cast<fftwf_complex*>(
            fftwf_malloc(sizeof(fftwf_complex) * totalSamples));

        transposed_data = reinterpret_cast<fftwf_complex*>(
            fftwf_malloc(sizeof(fftwf_complex) * totalSamples));

        doppler_fft_output = reinterpret_cast<fftwf_complex*>(
            fftwf_malloc(sizeof(fftwf_complex) * totalSamples));

        magnitude.resize(totalSamples);

        initialise_plans();
    }

    ~RangeDopplerProcessor()
    {
        // Clean up FFTW plans and free allocated memory
        fftwf_destroy_plan(range_plan);
        fftwf_destroy_plan(doppler_plan);

        fftwf_free(iq_data);
        fftwf_free(range_fft_output);
        fftwf_free(transposed_data);
        fftwf_free(doppler_fft_output);

        fftwf_cleanup_threads();
    }

    /**
     * @brief Generates synthetic IQ data for a single target with specified range and Doppler frequencies.
     * The IQ data is generated using a simple model where the phase of the signal is determined by the range and Doppler frequencies,
     * creating a sinusoidal pattern across both dimensions.
     * @param range_freq The frequency component corresponding to the target's range (fast time)
     * @param doppler_freq The frequency component corresponding to the target's Doppler shift (slow time)
     */
    void generate_iq_data(
        float range_freq,
        float doppler_freq)
    {
        #pragma omp parallel for collapse(2)
        for (int p = 0; p < pulses; ++p)
        {
            for (int s = 0; s < samplesPerPulse; ++s)
            {
                float phase =
                    2.0f * PI *
                    (
                        range_freq * static_cast<float>(s) +
                        doppler_freq * static_cast<float>(p)
                    );

                int idx = p * samplesPerPulse + s;

                iq_data[idx][0] = std::cos(phase);
                iq_data[idx][1] = std::sin(phase);
            }
        }
    }

    /**
     * @brief Processes the IQ data through the range and Doppler FFTs.
     */
    void process()
    {
        fftwf_execute(range_plan);

        transpose();

        fftwf_execute(doppler_plan);

        compute_magnitude();
    }

    /**
     * @brief Benchmarks the processing time of the range-Doppler algorithm by running it for a specified number of iterations
     * and measuring the average execution time in milliseconds.
     * @param iterations The number of times to run the process function for benchmarking (default is 100)
     */
    void benchmark(int iterations = 100)
    {
        auto start =
            std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            process();
        }

        auto end =
            std::chrono::high_resolution_clock::now();

        double ms =
            std::chrono::duration_cast<
                std::chrono::milliseconds>(
                    end - start)
                        .count();

        std::cout
            << "Average CPU Range-Doppler Time: "
            << ms / iterations
            << " ms\n";
    }

private:
    /**
     * @brief Initialises the FFTW plans for both the range and Doppler FFTs.
     * The plans are created using FFTW_MEASURE which provides good optimisation
     * without extremely long planning times.
     */
    void initialise_plans()
    {
        fftwf_init_threads();

        fftwf_plan_with_nthreads(4);

        /*
         * RANGE FFT
         */

        int range_rank = 1;
        int range_n[] = { samplesPerPulse };

        range_plan = fftwf_plan_many_dft(
            range_rank,
            range_n,
            pulses,
            iq_data,
            nullptr,
            1,
            samplesPerPulse,
            range_fft_output,
            nullptr,
            1,
            samplesPerPulse,
            FFTW_FORWARD,
            FFTW_MEASURE);

        /*
         * DOPPLER FFT
         */

        int doppler_rank = 1;
        int doppler_n[] = { pulses };

        doppler_plan = fftwf_plan_many_dft(
            doppler_rank,
            doppler_n,
            samplesPerPulse,
            transposed_data,
            nullptr,
            1,
            pulses,
            doppler_fft_output,
            nullptr,
            1,
            pulses,
            FFTW_FORWARD,
            FFTW_MEASURE);
    }

    /**
     * @brief Transposes the range FFT output so the Doppler FFT
     * can operate on contiguous memory.
     */
    void transpose()
    {
        #pragma omp parallel for collapse(2)
        for (int p = 0; p < pulses; ++p)
        {
            for (int s = 0; s < samplesPerPulse; ++s)
            {
                int src =
                    p * samplesPerPulse + s;

                int dst =
                    s * pulses + p;

                transposed_data[dst][0] =
                    range_fft_output[src][0];

                transposed_data[dst][1] =
                    range_fft_output[src][1];
            }
        }
    }

    /**
     * @brief Computes the magnitude of the range-Doppler map from the
     * complex Doppler FFT output.
     */
    void compute_magnitude()
    {
        #pragma omp parallel for simd
        for (int i = 0; i < totalSamples; ++i)
        {
            float re =
                doppler_fft_output[i][0];

            float im =
                doppler_fft_output[i][1];

            magnitude[i] =
                re * re + im * im;
        }
    }

public:
    /**
     * @brief Exports the computed range-Doppler map to a binary file.
     */
    void export_range_doppler_map_binary()
    {
        std::ofstream file(
            "range_doppler_map.bin",
            std::ios::binary);

        if (!file.is_open())
        {
            std::cerr
                << "Failed to open binary output file\n";

            return;
        }

        file.write(
            reinterpret_cast<char*>(magnitude.data()),
            magnitude.size() * sizeof(float));

        file.close();

        std::cout
            << "Binary range-Doppler map exported\n";
    }

private:
    int pulses;
    int samplesPerPulse;
    int totalSamples;

    fftwf_complex* iq_data{};
    fftwf_complex* range_fft_output{};
    fftwf_complex* transposed_data{};
    fftwf_complex* doppler_fft_output{};

    std::vector<float> magnitude;

    fftwf_plan range_plan{};
    fftwf_plan doppler_plan{};
};

int main(int argc, char* argv[])
{
    /**
     * Usage:
     *
     * ./range-profile-cpu
     *      <pulses>
     *      <samples>
     *      <benchmark|export>
     */

    if (argc < 4)
    {
        std::cerr
            << "Usage:\n"
            << argv[0]
            << " <pulses> <samples> <mode>\n";

        return -1;
    }

    const int pulses =
        std::stoi(argv[1]);

    const int samples =
        std::stoi(argv[2]);

    std::string mode =
        argv[3];

    constexpr float range_freq = 0.08f;
    constexpr float doppler_freq = 0.12f;

    RangeDopplerProcessor processor(
        pulses,
        samples);

    processor.generate_iq_data(
        range_freq,
        doppler_freq);

    if (mode == "benchmark")
    {
        processor.benchmark();
    }
    else if (mode == "export")
    {
        auto start =
            std::chrono::high_resolution_clock::now();

        processor.process();

        auto end =
            std::chrono::high_resolution_clock::now();

        double ms =
            std::chrono::duration_cast<
                std::chrono::milliseconds>(
                    end - start)
                        .count();

        std::cout
            << "CPU Range-Doppler Time: "
            << ms
            << " ms\n";

        processor.export_range_doppler_map_binary();
    }
    else
    {
        std::cerr
            << "Invalid mode. Use 'benchmark' or 'export'.\n";

        return -1;
    }

    return 0;
}