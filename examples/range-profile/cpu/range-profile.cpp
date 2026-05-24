#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include <vector>

constexpr float PI = 3.14159265358979323846f;

using Complex = std::complex<float>;

struct Detection
{
    int bin;
    float power;
};

struct ScopedTimer
{
    std::string name;
    std::chrono::high_resolution_clock::time_point start;

    ScopedTimer(const std::string& name) : name(name), start(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << name << " took " << duration << " ms\n";
    }
};

std::vector<Complex> generateIQData(int n, int targetBin, float amplitude, float noiseStdDev)
{
    ScopedTimer timer("generateIQData");
    std::vector<Complex> iqData(n);
    std::default_random_engine rng(42); // Fixed seed for reproducibility
    std::normal_distribution<float> noiseDist(0.0f, noiseStdDev);

    for (int i = 0; i < n; ++i)
    {
        float phase = PI * targetBin * static_cast<float>(i) / static_cast<float>(n); // Simulate a target at the specified bin
        Complex Target = amplitude * Complex(std::cos(phase), std::sin(phase)); // Target signal
        Complex Noise = Complex(noiseDist(rng), noiseDist(rng)); // Additive
        iqData[i] = Target + Noise; // Combine target and noise
    }

    return iqData;
}

void applyHannWindow(std::vector<Complex>& data)
{
    ScopedTimer timer("applyHannWindow");
    int n = data.size();
    for (int i = 0; i < n; ++i)
    {
        float hannValue = 0.5f * (1.0f - std::cos(2.0f * PI * static_cast<float>(i) / static_cast<float>(n - 1)));
        data[i] *= hannValue;
    }
}

std::vector<Complex> naive_dft(const std::vector<Complex>& input)
{
    ScopedTimer timer("naive_dft");
    int n = input.size();
    std::vector<Complex> output(n);
    for (int k = 0; k < n; ++k)
    {
        Complex sum(0.0f, 0.0f);
        for (int j = 0; j < n; ++j)
        {
            float angle = -2.0f * PI * static_cast<float>(k) * static_cast<float>(j) / static_cast<float>(n);
            sum += input[j] * Complex(std::cos(angle), std::sin(angle));
        }
        output[k] = sum;
    }
    return output;
}

std::vector<float> computePower(const std::vector<Complex>& data)
{
    ScopedTimer timer("computePower");
    std::vector<float> power(data.size());
    for (size_t i = 0; i < power.size(); ++i)
    {
        float real = data[i].real();
        float imag = data[i].imag();

        power[i] = real * real + imag * imag; // Power is the magnitude squared of the complex number
    }
    return power;
}

std::vector<Detection> detectTargets(const std::vector<float>& power, float threshold)
{
    ScopedTimer timer("detectTargets");
    std::vector<Detection> detections;
    for (size_t i = 0; i < power.size(); ++i)
    {
        if (power[i] > threshold)
        {
            detections.push_back({static_cast<int>(i), power[i]});
        }
    }
    return detections;
}
void log_signal_summary(
    const std::vector<Complex>& signal,
    const std::string& name,
    int sampleCount)
{
    std::cout << "\n" << name << " summary\n";
    std::cout << "Size: " << signal.size() << " samples\n";

    int count = std::min(sampleCount, static_cast<int>(signal.size()));

    std::cout << "First " << count << " samples:\n";

    for (int i = 0; i < count; ++i)
    {
        std::cout << "  [" << i << "] "
                  << "real=" << signal[i].real()
                  << ", imag=" << signal[i].imag()
                  << ", mag=" << std::abs(signal[i])
                  << "\n";
    }
}

void log_power_summary(
    const std::vector<float>& power,
    int binsAroundPeak)
{
    auto maxIt = std::max_element(power.begin(), power.end());

    int peakBin = static_cast<int>(std::distance(power.begin(), maxIt));
    float peakPower = *maxIt;

    std::cout << "\nPower spectrum summary\n";
    std::cout << "Number of bins: " << power.size() << "\n";
    std::cout << "Peak bin: " << peakBin << "\n";
    std::cout << "Peak power: " << peakPower << "\n";
    //std::cout << "Peak power dB: " << to_db(peakPower) << " dB\n";

    int start = std::max(0, peakBin - binsAroundPeak);
    int end = std::min(static_cast<int>(power.size()) - 1, peakBin + binsAroundPeak);

    std::cout << "\nBins around peak:\n";

    for (int i = start; i <= end; ++i)
    {
        std::cout << "  Bin " << i
                  << ": power=" << power[i]
                  //<< ", dB=" << to_db(power[i])
                  << "\n";
    }
}

void log_detections(const std::vector<Detection>& detections)
{
    std::cout << "\nDetection summary\n";
    std::cout << "Number of detections: " << detections.size() << "\n";

    if (detections.empty())
    {
        std::cout << "No detections above threshold.\n";
        return;
    }

    for (const auto& detection : detections)
    {
        std::cout << "  Bin " << detection.bin
                  << ": power=" << detection.power
                  //<< ", dB=" << to_db(detection.power)
                  << "\n";
    }
}

int main()
{
    constexpr int N = 1024;

    constexpr float targetBin = 128.0f;
    constexpr float amplitude = 1.0f;
    constexpr float noiseStdDev = 0.2f;

    constexpr float threshold = 10000.0f;

    std::cout << "Radar CPU baseline\n";
    std::cout << "==================\n";

    std::cout << "\nConfiguration\n";
    std::cout << "Samples: " << N << "\n";
    std::cout << "Synthetic target bin: " << targetBin << "\n";
    std::cout << "Target amplitude: " << amplitude << "\n";
    std::cout << "Noise std dev: " << noiseStdDev << "\n";
    std::cout << "Detection threshold: " << threshold << "\n";

    std::cout << "\nGenerating synthetic IQ signal...\n";
    auto signal = generateIQData(
        N,
        targetBin,
        amplitude,
        noiseStdDev);

    log_signal_summary(signal, "Raw IQ signal", 5);

    std::cout << "\nApplying Hann window...\n";
    auto windowedSignal = signal;
    applyHannWindow(windowedSignal);

    log_signal_summary(windowedSignal, "Windowed IQ signal", 5);

    std::cout << "\nRunning naive DFT...\n";
    auto spectrum = naive_dft(windowedSignal);

    std::cout << "Computing power spectrum...\n";
    auto power = computePower(spectrum);

    log_power_summary(power, 5);

    std::cout << "\nRunning threshold detector...\n";
    auto detections = detectTargets(power, threshold);

    log_detections(detections);

    std::cout << "\nExpected target bin: " << targetBin << "\n";

    if (!power.empty())
    {
        auto maxIt = std::max_element(power.begin(), power.end());
        int peakBin = static_cast<int>(std::distance(power.begin(), maxIt));

        std::cout << "Measured peak bin: " << peakBin << "\n";
        std::cout << "Bin error: " << std::abs(peakBin - static_cast<int>(targetBin)) << "\n";
    }

    std::cout << "\nDone.\n";

    return 0;
}