#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <thread>
#include <fstream>

#include <fftw3.h>

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
     * will preallocate memory for the input IQ data, intermediate FFT outputs, and the final magnitude array
     * It also initializes the FFTW plans for both the range and Doppler FFTs.
     * @param pulses The number of radar pulses (slow time dimension)
     * @param samplesPerPulse The number of samples per pulse (fast time dimension)
     */
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
        // Clean up FFTW plans and free allocated memory
        fftwf_destroy_plan(range_plan);
        fftwf_destroy_plan(doppler_plan);

        // Free the allocated memory for FFTW arrays
        fftwf_free(iq_data);
        fftwf_free(range_fft_output);
        fftwf_free(doppler_fft_output);
        fftwf_free(transposed_data);
        fftwf_free(transposed_output);

        fftwf_cleanup_threads();
    }

    /**
     * @brief Generates synthetic IQ data for a single target with specified range and Doppler frequencies.
     * The IQ data is generated using a simple model where the phase of the signal is determined by the range and Doppler frequencies, creating a sinusoidal pattern across both dimensions.
     * @param range_freq The frequency component corresponding to the target's range (fast time)
     * @param doppler_freq The frequency component corresponding to the target's Doppler shift (slow time)
     */
    void generate_iq_data(float range_freq, float doppler_freq)
    {
        // For each pulse and sample, compute the phase based on the range and Doppler frequencies
        for (int p = 0; p < pulses; ++p)
        {
            for (int s = 0; s < samplesPerPulse; ++s)
            {
                // The phase is a combination of the range frequency (which varies with sample index) 
                // and the Doppler frequency (which varies with pulse index)
                float phase =
                    2.0f * PI *
                    (range_freq * static_cast<float>(s) +
                     doppler_freq * static_cast<float>(p));

                int idx = p * samplesPerPulse + s;

                // Convert the phase to IQ components using cosine for the real part and sine for the imaginary part
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

        transpose_back();

        compute_magnitude();
    }

    /**
     * @brief Benchmarks the processing time of the range-Doppler algorithm by running it for a specified number of iterations
     * and measuring the average execution time in milliseconds.
     * @param iterations The number of times to run the process function for benchmarking (default is 100)
     */
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
    /**
     * @brief Initialises the FFTW plans for both the range and Doppler FFTs. 
     * The plans are created using the FFTW_MEASURE flag, which allows FFTW to optimize the execution plan based on the input size
     * and hardware capabilities. The range FFT is planned to operate across the samples for each pulse, while the Doppler FFT is 
     * planned to operate across the pulses after transposing the data. Multithreading is enabled to take advantage of multiple CPU 
     * cores for faster execution.
     */
    void initialise_plans()
    {
        fftwf_init_threads();

        fftwf_plan_with_nthreads(
            std::thread::hardware_concurrency());
        
        // Range FFT
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

        // Doppler FFT
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

    /**
     * @brief Function to transpose the range FFT output so that it can be used as input for the Doppler FFT. This is to enable
     * row-major access patterns for the Doppler FFT, which operates across the pulses. 
     * The transposition rearranges the data from a format where each pulse's samples are contiguous to a format where each 
     * sample's pulses are contiguous.
     */
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

    /**
     * @brief Function to transpose the Doppler FFT output back to the original format. After the Doppler FFT is performed
     * on the transposed data,
     */
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

    /**
     * @brief Computes the magnitude of the range-Doppler map from the complex Doppler FFT output.
     * The magnitude is calculated as the sum of squares of the real and imaginary parts for each complex value in the Doppler FFT output.
     * This results in a real-valued array representing the power of the signal at each range and Doppler bin, which can be used for 
     * further analysis or visualization.
     */
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
    /**
     * @brief Exports the computed range-Doppler map to a CSV file. The output is formatted as a dense matrix where rows 
     * correspond to Doppler bins and columns correspond to range bins.
     */
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