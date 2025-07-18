using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using UnityEngine;

public class GtcrnTest3 : MonoBehaviour
{
    private InferenceSession _onnxSession;
    private string _modelPath;
    private readonly SessionOptions _sessionOptions;

    // STFT parameters
    private const int N_FFT = 512;
    private const int HOP_LENGTH = 256;
    private const int WIN_LENGTH = 512;
    private const int SAMPLE_RATE = 16000;

    // Cache dimensions (adjust based on your model architecture)
    private const int CONV_CACHE_DIM1 = 2;
    private const int CONV_CACHE_DIM2 = 1;
    private const int CONV_CACHE_DIM3 = 16;
    private const int CONV_CACHE_DIM4 = 16;
    private const int CONV_CACHE_DIM5 = 33;

    private const int TRA_CACHE_DIM1 = 2;
    private const int TRA_CACHE_DIM2 = 3;
    private const int TRA_CACHE_DIM3 = 1;
    private const int TRA_CACHE_DIM4 = 1;
    private const int TRA_CACHE_DIM5 = 16;

    private const int INTER_CACHE_DIM1 = 2;
    private const int INTER_CACHE_DIM2 = 1;
    private const int INTER_CACHE_DIM3 = 33;
    private const int INTER_CACHE_DIM4 = 16;

    // Start is called before the first frame update
    void Start()
    {
        _modelPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        _onnxSession = new InferenceSession(_modelPath);

        StreamingInference(Application.dataPath + "/mix.wav", Application.dataPath + "/result.wav");
    }

    // Update is called once per frame
    void Update()
    {

    }

    /// <summary>
    /// Perform streaming inference frame by frame
    /// </summary>
    public void StreamingInference(string audioPath, string outputPath = null)
    {
        var audio = Util.ReadWav(audioPath);
        var stft = STFT(audio);

        var numBins = stft.GetLength(0);
        var numFrames = stft.GetLength(1);

        // Initialize caches
        var (convCache, traCache, interCache) = InitializeCaches();

        var enhancedSTFT = new Complex[numBins, numFrames];
        var timings = new List<double>();

        Console.WriteLine($"Processing {numFrames} frames...");

        for (int frame = 0; frame < numFrames; frame++)
        {
            // Prepare input tensor for current frame
            var inputTensor = new DenseTensor<float>(new[] { 1, numBins, 1, 2 });

            for (int bin = 0; bin < numBins; bin++)
            {
                inputTensor[0, bin, 0, 0] = (float)stft[bin, frame].Real;
                inputTensor[0, bin, 0, 1] = (float)stft[bin, frame].Imaginary;
            }

            var stopwatch = Stopwatch.StartNew();

            // Run inference
            var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("mix", inputTensor),
                    NamedOnnxValue.CreateFromTensor("conv_cache", convCache),
                    NamedOnnxValue.CreateFromTensor("tra_cache", traCache),
                    NamedOnnxValue.CreateFromTensor("inter_cache", interCache)
                };

            using var results = _onnxSession.Run(inputs);

            stopwatch.Stop();
            timings.Add(stopwatch.Elapsed.TotalMilliseconds);

            // Extract outputs
            var enhancedFrame = results.First(r => r.Name == "enh").AsTensor<float>();
            convCache = results.First(r => r.Name == "conv_cache_out").AsTensor<float>().ToDenseTensor();
            traCache = results.First(r => r.Name == "tra_cache_out").AsTensor<float>().ToDenseTensor();
            interCache = results.First(r => r.Name == "inter_cache_out").AsTensor<float>().ToDenseTensor();

            // Store enhanced frame
            for (int bin = 0; bin < numBins; bin++)
            {
                enhancedSTFT[bin, frame] = new Complex(
                    enhancedFrame[0, bin, 0, 0],
                    enhancedFrame[0, bin, 0, 1]
                );
            }

            if (frame % 100 == 0)
            {
                Console.WriteLine($"Processed frame {frame}/{numFrames}");
            }
        }

        // Convert back to time domain
        var enhancedAudio = ISTFT(enhancedSTFT);

        // Save output if path provided
        if (!string.IsNullOrEmpty(outputPath))
        {
            Util.SaveClip(1, 16000, enhancedAudio, outputPath);
        }

        // Print timing statistics
        if (timings.Count > 0)
        {
            Console.WriteLine($"Streaming inference complete.");
            Console.WriteLine($"Timing stats - Mean: {timings.Average():F1}ms, " +
                            $"Max: {timings.Max():F1}ms, Min: {timings.Min():F1}ms");
        }
    }

    /// <summary>
    /// Initialize streaming caches with proper dimensions
    /// </summary>
    public (DenseTensor<float> convCache, DenseTensor<float> traCache, DenseTensor<float> interCache) InitializeCaches()
    {
        var convCache = new DenseTensor<float>(new[] { CONV_CACHE_DIM1, CONV_CACHE_DIM2, CONV_CACHE_DIM3, CONV_CACHE_DIM4, CONV_CACHE_DIM5 });
        var traCache = new DenseTensor<float>(new[] { TRA_CACHE_DIM1, TRA_CACHE_DIM2, TRA_CACHE_DIM3, TRA_CACHE_DIM4, TRA_CACHE_DIM5 });
        var interCache = new DenseTensor<float>(new[] { INTER_CACHE_DIM1, INTER_CACHE_DIM2, INTER_CACHE_DIM3, INTER_CACHE_DIM4 });

        // Initialize with zeros
        convCache.Fill(0.0f);
        traCache.Fill(0.0f);
        interCache.Fill(0.0f);

        return (convCache, traCache, interCache);
    }

    /// <summary>
    /// Perform Short-Time Fourier Transform
    /// </summary>
    public Complex[,] STFT(float[] audio)
    {
        var windowFunc = CreateHannWindow(WIN_LENGTH);
        var numFrames = (audio.Length - WIN_LENGTH) / HOP_LENGTH + 1;
        var numBins = N_FFT / 2 + 1;

        var stftResult = new Complex[numBins, numFrames];
        var fftBuffer = new Complex[N_FFT];

        for (int frame = 0; frame < numFrames; frame++)
        {
            var frameStart = frame * HOP_LENGTH;

            // Apply window and copy to FFT buffer
            for (int i = 0; i < WIN_LENGTH; i++)
            {
                if (frameStart + i < audio.Length)
                {
                    fftBuffer[i] = new Complex(audio[frameStart + i] * windowFunc[i], 0);
                }
                else
                {
                    fftBuffer[i] = Complex.Zero;
                }
            }

            // Zero pad
            for (int i = WIN_LENGTH; i < N_FFT; i++)
            {
                fftBuffer[i] = Complex.Zero;
            }

            // Perform FFT
            FFT(fftBuffer, N_FFT);

            // Copy positive frequencies
            for (int bin = 0; bin < numBins; bin++)
            {
                stftResult[bin, frame] = fftBuffer[bin];
            }
        }

        return stftResult;
    }

    /// <summary>
    /// Perform Inverse Short-Time Fourier Transform
    /// </summary>
    public float[] ISTFT(Complex[,] stft)
    {
        var numBins = stft.GetLength(0);
        var numFrames = stft.GetLength(1);
        var windowFunc = CreateHannWindow(WIN_LENGTH);

        var outputLength = (numFrames - 1) * HOP_LENGTH + WIN_LENGTH;
        var output = new float[outputLength];
        var windowSum = new float[outputLength];

        var ifftBuffer = new Complex[N_FFT];

        for (int frame = 0; frame < numFrames; frame++)
        {
            // Prepare IFFT input
            for (int bin = 0; bin < numBins; bin++)
            {
                ifftBuffer[bin] = stft[bin, frame];
            }

            // Mirror for negative frequencies
            for (int bin = 1; bin < numBins - 1; bin++)
            {
                ifftBuffer[N_FFT - bin] = Complex.Conjugate(ifftBuffer[bin]);
            }

            // Perform IFFT
            IFFT(ifftBuffer, N_FFT);

            // Overlap-add
            var frameStart = frame * HOP_LENGTH;
            for (int i = 0; i < WIN_LENGTH; i++)
            {
                if (frameStart + i < outputLength)
                {
                    output[frameStart + i] += (float)ifftBuffer[i].Real * windowFunc[i];
                    windowSum[frameStart + i] += windowFunc[i] * windowFunc[i];
                }
            }
        }

        // Normalize
        for (int i = 0; i < outputLength; i++)
        {
            if (windowSum[i] > 1e-10f)
            {
                output[i] /= windowSum[i];
            }
        }

        return output;
    }

    /// <summary>
    /// Create Hann window
    /// </summary>
    private float[] CreateHannWindow(int length)
    {
        var window = new float[length];
        for (int i = 0; i < length; i++)
        {
            window[i] = (float)(0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1))));
        }

        // Apply square root as in the original code
        for (int i = 0; i < length; i++)
        {
            window[i] = (float)Math.Sqrt(window[i]);
        }

        return window;
    }

    /// <summary>
    /// Fast Fourier Transform
    /// </summary>
    private void FFT(Complex[] buffer, int length)
    {
        var j = 0;
        for (var i = 0; i < length - 1; i++)
        {
            if (i < j)
            {
                var temp = buffer[i];
                buffer[i] = buffer[j];
                buffer[j] = temp;
            }

            var k = length >> 1;
            while (j >= k)
            {
                j -= k;
                k >>= 1;
            }
            j += k;
        }

        for (var length2 = 2; length2 <= length; length2 <<= 1)
        {
            var w = Complex.Exp(-Complex.ImaginaryOne * Math.PI / (length2 >> 1));
            for (var i = 0; i < length; i += length2)
            {
                var wn = Complex.One;
                for (var j1 = 0; j1 < (length2 >> 1); j1++)
                {
                    var u = buffer[i + j1];
                    var v = buffer[i + j1 + (length2 >> 1)] * wn;
                    buffer[i + j1] = u + v;
                    buffer[i + j1 + (length2 >> 1)] = u - v;
                    wn *= w;
                }
            }
        }
    }

    /// <summary>
    /// Inverse Fast Fourier Transform
    /// </summary>
    private void IFFT(Complex[] buffer, int length)
    {
        // Conjugate input
        for (int i = 0; i < length; i++)
        {
            buffer[i] = Complex.Conjugate(buffer[i]);
        }

        // Perform FFT
        FFT(buffer, length);

        // Conjugate output and scale
        for (int i = 0; i < length; i++)
        {
            buffer[i] = Complex.Conjugate(buffer[i]) / length;
        }
    }
}