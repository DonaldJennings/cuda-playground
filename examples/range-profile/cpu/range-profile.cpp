#include <iostream>
#include <vector>
#include <complex>
#include <chrono>

#include <fftw3.h>

class RangeDopplerProcessor
{
    private:
        using Complex = std::complex<float>;
        constexpr static float PI = 3.1415926538979323846f;
    public:
        RangeDopplerProcessor(int pulses, int samplesPerPulse)
            : pulses{pulses}, samplesPerPulse{samplesPerPulse}
            {
                std::cout << "Constructing RangeDopplerProcessor with " << std::to_string(pulses * samplesPerPulse) << " samples\n";
                iq_data.resize(pulses * samplesPerPulse);
                range_fft_output.resize(pulses * samplesPerPulse);
                doppler_fft_output.resize(pulses * samplesPerPulse);
                magnitude.resize(pulses * samplesPerPulse);
            }
        
        void generate_iq_data(float range_freq, float doppler_freq)
        {
            for (int p=0; p < pulses; ++p)
            {
                for (int s=0; s < samplesPerPulse; ++s)
                {
                    float phase = 2.0f * PI * (range_freq * s + doppler_freq * p);
                    iq_data[p * samplesPerPulse + s] = Complex(std::cos(phase), std::sin(phase));
                }
            }
        }

        void process()
        {
            perform_range_fft();
            perform_doppler_fft();
            compute_magnitude();
        }

        void benchmark()
        {
            auto start { std::chrono::high_resolution_clock::now()};
            process();
            auto end{ std::chrono::high_resolution_clock::now()};

            double ms = std::chrono::duration<double, std::milli>(end-start).count();
            std::cout << "CPU Range-Doppler Time: " << ms << " ms\n";
        }

    private:
        void perform_doppler_fft()
        {
            std::vector<Complex> temp(pulses);
            fftwf_complex* in = reinterpret_cast<fftwf_complex*>(temp.data());
            fftwf_complex* out = reinterpret_cast<fftwf_complex*>(temp.data());

            fftwf_plan plan = fftwf_plan_dft_1d(
                pulses,
                in,
                out,
                FFTW_FORWARD,
                FFTW_ESTIMATE
            );

            for (int range_bin=0; range_bin < samplesPerPulse; ++range_bin)
            {
                for (int pulse=0; pulse < pulses; ++pulse)
                {
                    temp[pulse] = range_fft_output[pulse * samplesPerPulse + range_bin];
                }

                fftwf_execute(plan);

                for (int pulse=0; pulse < pulses; ++pulse)
                {
                    doppler_fft_output[pulse * samplesPerPulse + range_bin] = temp[pulse];
                }
            }

            fftwf_destroy_plan(plan);
        }

        void perform_range_fft()
        {
            fftwf_complex* in = reinterpret_cast<fftwf_complex*>(iq_data.data());
            fftwf_complex* out = reinterpret_cast<fftwf_complex*>(range_fft_output.data());
            
            for (int pulse=0; pulse < pulses; ++pulses)
            {
                fftwf_plan plan = fftwf_plan_dft_1d(
                    samplesPerPulse,
                    in + pulse * samplesPerPulse,
                    out + pulse * samplesPerPulse,
                    FFTW_FORWARD,
                    FFTW_ESTIMATE
                );

                fftwf_execute(plan);
                fftwf_destroy_plan(plan);
            }
        }


        void compute_magnitude()
        {
            for (std::size_t i=0; i < doppler_fft_output.size(); ++i)
            {
                magnitude[i] = std::norm(doppler_fft_output[i]);
            }
        }

    private:
        int pulses;
        int samplesPerPulse;
        std::vector<Complex> iq_data;
        std::vector<Complex> range_fft_output;
        std::vector<Complex> doppler_fft_output;
        std::vector<float> magnitude;
};

int main()
{
    constexpr int pulses = 128;
    constexpr int samples = 1024;
    constexpr float range_freq = 0.05f;
    constexpr float doppler_freq = 0.02f;

    RangeDopplerProcessor processor(pulses, samples);

    processor.generate_iq_data(range_freq, doppler_freq);

    processor.benchmark();
    return 0;
}