#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <thread>
#include <fstream>

#include <fftw3.h>

class RangeDopplerProcessor
{
private:
    constexpr static float PI = 3.14159265358979323846f;

public:
    RangeDopplerProcessor(int pulses, int samplesPerPulse)
        : pulses(pulses),
          samplesPerPulse(samplesPerPulse),
          totalSamples(pulses * samplesPerPulse)
    {
        std::cout << "Allocating "
                  << totalSamples
                  << " complex samples\n";

        iq_data = reinterpret_cast<fftwf_complex*>(
            fftwf_malloc(sizeof(fftwf_complex) * totalSamples));

        range_fft_output = reinterpret_cast<fftwf_complex*>(
            fftwf_malloc(sizeof(fftwf_complex) * totalSamples));

        doppler_fft_output = reinterpret_cast<fftwf_complex*>(
            fftwf_malloc(sizeof(fftwf_complex) * totalSamples));

        transposed_data = reinterpret_cast<fftwf_complex*>(
            fftwf_malloc(sizeof(fftwf_complex) * totalSamples));

        transposed_output = reinterpret_cast<fftwf_complex*>(
            fftwf_malloc(sizeof(fftwf_complex) * totalSamples));

        magnitude.resize(totalSamples);

        initialise_plans();
    }

    ~RangeDopplerProcessor()
    {
        fftwf_destroy_plan(range_plan);
        fftwf_destroy_plan(doppler_plan);

        fftwf_free(iq_data);
        fftwf_free(range_fft_output);
        fftwf_free(doppler_fft_output);
        fftwf_free(transposed_data);
        fftwf_free(transposed_output);

        fftwf_cleanup_threads();
    }

    void generate_iq_data(float range_freq, float doppler_freq)
    {
        for (int p = 0; p < pulses; ++p)
        {
            for (int s = 0; s < samplesPerPulse; ++s)
            {
                float phase =
                    2.0f * PI *
                    (range_freq * static_cast<float>(s) +
                     doppler_freq * static_cast<float>(p));

                int idx = p * samplesPerPulse + s;

                iq_data[idx][0] = std::cos(phase);
                iq_data[idx][1] = std::sin(phase);
            }
        }
    }

    void process()
    {
        fftwf_execute(range_plan);

        transpose();

        fftwf_execute(doppler_plan);

        transpose_back();

        compute_magnitude();
    }

    void benchmark(int iterations = 100)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            process();
        }

        auto end = std::chrono::high_resolution_clock::now();

        double ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start)
                .count();

        std::cout << "Average CPU Range-Doppler Time: "
                  << ms / iterations
                  << " ms\n";
    }

private:
    void initialise_plans()
    {
        fftwf_init_threads();

        fftwf_plan_with_nthreads(
            std::thread::hardware_concurrency());

        /*
         * RANGE FFT
         * FFT across samples for each pulse
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
         * FFT across pulses after transpose
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
            transposed_output,
            nullptr,
            1,
            pulses,
            FFTW_FORWARD,
            FFTW_MEASURE);
    }

    void transpose()
    {
        for (int p = 0; p < pulses; ++p)
        {
            for (int s = 0; s < samplesPerPulse; ++s)
            {
                int src = p * samplesPerPulse + s;
                int dst = s * pulses + p;

                transposed_data[dst][0] =
                    range_fft_output[src][0];

                transposed_data[dst][1] =
                    range_fft_output[src][1];
            }
        }
    }

    void transpose_back()
    {
        for (int s = 0; s < samplesPerPulse; ++s)
        {
            for (int p = 0; p < pulses; ++p)
            {
                int src = s * pulses + p;
                int dst = p * samplesPerPulse + s;

                doppler_fft_output[dst][0] =
                    transposed_output[src][0];

                doppler_fft_output[dst][1] =
                    transposed_output[src][1];
            }
        }
    }

    void compute_magnitude()
    {
        for (int i = 0; i < totalSamples; ++i)
        {
            float re = doppler_fft_output[i][0];
            float im = doppler_fft_output[i][1];

            magnitude[i] = re * re + im * im;
        }
    }

    public:
void export_range_doppler_map()
{
    std::ofstream file("range_doppler_map.csv");

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file for writing\n";
        return;
    }

    /*
     * Export as dense matrix:
     *
     * Rows    -> Doppler bins
     * Columns -> Range bins
     */

    for (int p = 0; p < pulses; ++p)
    {
        for (int s = 0; s < samplesPerPulse; ++s)
        {
            int idx = p * samplesPerPulse + s;

            file << magnitude[idx];

            if (s != samplesPerPulse - 1)
            {
                file << ",";
            }
        }

        file << "\n";
    }

    file.close();

    std::cout << "Range-Doppler map exported to "
              << "range_doppler_map.csv\n";
}

private:
    int pulses;
    int samplesPerPulse;
    int totalSamples;

    fftwf_complex* iq_data{};
    fftwf_complex* range_fft_output{};
    fftwf_complex* doppler_fft_output{};

    fftwf_complex* transposed_data{};
    fftwf_complex* transposed_output{};

    std::vector<float> magnitude;

    fftwf_plan range_plan{};
    fftwf_plan doppler_plan{};
};

int main()
{
    constexpr int pulses = 128;
    constexpr int samples = 1024;

    constexpr float range_freq = 0.08f;
    constexpr float doppler_freq = 0.12f;

    RangeDopplerProcessor processor(
        pulses,
        samples);

    processor.generate_iq_data(
        range_freq,
        doppler_freq);

    processor.benchmark();

    processor.export_range_doppler_map();

    return 0;
}