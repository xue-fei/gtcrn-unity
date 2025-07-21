using System;
using System.Linq;
using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;

public class GtcrnStream : IDisposable
{
    private InferenceSession onnxSession;
    // STFT parameters
    private const int N_FFT = 512;
    private const int HOP_LENGTH = 256;
    private const int WIN_LENGTH = 512;
    private const int SAMPLE_RATE = 16000;

    public GtcrnStream(string modelPath)
    {
        onnxSession = new InferenceSession(modelPath);
        UnityEngine.Debug.Log("init done");
    }

    public float[] ProcessFrame(float[] audioFrame)
    {
        Complex[,] stft = STFT(audioFrame);
        int numBins = stft.GetLength(0);
        int numFrames = stft.GetLength(1);
        UnityEngine.Debug.Log("numFrames:"+ numFrames);

        DenseTensor<float> convCache = new DenseTensor<float>(new[] { 2, 1, 16, 16, 33 });
        DenseTensor<float> traCache = new DenseTensor<float>(new[] { 2, 3, 1, 1, 16 });
        DenseTensor<float> interCache = new DenseTensor<float>(new[] { 2, 1, 33, 16 });

        // Initialize with zeros
        convCache.Fill(0.0f);
        traCache.Fill(0.0f);
        interCache.Fill(0.0f);

        Complex[,] enhancedSTFT = new Complex[numBins, numFrames];

        for (int frame = 0; frame < numFrames; frame++)
        {
            // Prepare input tensor for current frame
            var inputTensor = new DenseTensor<float>(new[] { 1, numBins, 1, 2 });

            for (int bin = 0; bin < numBins; bin++)
            {
                inputTensor[0, bin, 0, 0] = (float)stft[bin, frame].Real;
                inputTensor[0, bin, 0, 1] = (float)stft[bin, frame].Imaginary;
            }

            // Run inference
            var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("mix", inputTensor),
                    NamedOnnxValue.CreateFromTensor("conv_cache", convCache),
                    NamedOnnxValue.CreateFromTensor("tra_cache", traCache),
                    NamedOnnxValue.CreateFromTensor("inter_cache", interCache)
                };

            using var results = onnxSession.Run(inputs);

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
        }

        return ISTFT(enhancedSTFT);
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

    public void Dispose()
    {
        onnxSession?.Dispose();
    }
}