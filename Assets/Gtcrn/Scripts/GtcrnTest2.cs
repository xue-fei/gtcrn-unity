using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;

public class GtcrnTest2 : MonoBehaviour
{
    string simpleModelPath;
    string outputPath;
    int sampleRate = 16000;
    int n_fft = 512;
    int hop_length = 256;
    int win_length = 512;

    // Start is called before the first frame update
    void Start()
    {
        simpleModelPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        outputPath = Application.streamingAssetsPath + "/result.wav";

        float[] rawAudio = Util.ReadWav(Application.streamingAssetsPath + "/mix.wav");
        float[] hanningWin = HanningWindow(win_length, 0.5f);
        (float[,,] x_stft, int numTimeFrames) = STFT(rawAudio, n_fft, hop_length, win_length, hanningWin);

        int numFreqBins = x_stft.GetLength(0);
        int numRealImagParts = x_stft.GetLength(2);
        int batchSize = 1;
        float[] y = new float[rawAudio.Length];
        Array.Copy(rawAudio, y, rawAudio.Length);

        var session = new InferenceSession(simpleModelPath);

        var convCacheTensor = new DenseTensor<float>(new[] { 2, 1, 16, 16, 33 });
        var traCacheTensor = new DenseTensor<float>(new[] { 2, 3, 1, 1, 16 });
        var interCacheTensor = new DenseTensor<float>(new[] { 2, 1, 33, 16 });
        convCacheTensor.Fill(0.0f);
        traCacheTensor.Fill(0.0f);
        interCacheTensor.Fill(0.0f);

        List<float[,,]> outputsList = new List<float[,,]>();
        List<double> T_list = new List<double>();
        Stopwatch stopwatch = new Stopwatch();

        for (int i = 0; i < numTimeFrames; i++)
        {
            stopwatch.Restart();

            var mixInputTensor = new DenseTensor<float>(new[] { batchSize, numFreqBins, 1, numRealImagParts });
            for (int b = 0; b < batchSize; b++)
            {
                for (int k = 0; k < numFreqBins; k++)
                {
                    for (int r_i = 0; r_i < numRealImagParts; r_i++)
                    {
                        mixInputTensor[b, k, 0, r_i] = x_stft[k, i, r_i];
                    }
                }
            }
            var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("mix", mixInputTensor),
                    NamedOnnxValue.CreateFromTensor("conv_cache", convCacheTensor),
                    NamedOnnxValue.CreateFromTensor("tra_cache", traCacheTensor),
                    NamedOnnxValue.CreateFromTensor("inter_cache", interCacheTensor)
                };

            using (var results = session.Run(inputs))
            {
                stopwatch.Stop();
                T_list.Add(stopwatch.Elapsed.TotalSeconds);
                var outITensor = results.First().AsTensor<float>();
                var currentOutI = new float[outITensor.Dimensions[1], outITensor.Dimensions[2], outITensor.Dimensions[3]];
                for (int k = 0; k < outITensor.Dimensions[1]; k++)
                {
                    for (int f = 0; f < outITensor.Dimensions[2]; f++)
                    {
                        for (int r_i = 0; r_i < outITensor.Dimensions[3]; r_i++)
                        {
                            currentOutI[k, f, r_i] = outITensor[0, k, f, r_i]; // Batch 0
                        }
                    }
                }
                outputsList.Add(currentOutI);

                convCacheTensor = (DenseTensor<float>)results[1].AsTensor<float>();
                traCacheTensor = (DenseTensor<float>)results[2].AsTensor<float>();
                interCacheTensor = (DenseTensor<float>)results[3].AsTensor<float>();
            }
        }

        int finalNumFreqBins = outputsList[0].GetLength(0);
        int finalNumRealImag = outputsList[0].GetLength(2);

        float[,,] finalOutputs = new float[finalNumFreqBins, numTimeFrames, finalNumRealImag];
        for (int i = 0; i < numTimeFrames; i++)
        {
            var currentOutI = outputsList[i];
            for (int k = 0; k < finalNumFreqBins; k++)
            {
                for (int r_i = 0; r_i < finalNumRealImag; r_i++)
                {
                    finalOutputs[k, i, r_i] = currentOutI[k, 0, r_i];
                }
            }
        }
        float[] enhanced = ISTFT(finalOutputs, n_fft, hop_length, win_length, hanningWin);

        Util.SaveClip(1, 16000, enhanced, outputPath);
    }

    // Update is called once per frame
    void Update()
    {

    }

    public static float[] HanningWindow(int length, float power)
    {
        float[] window = new float[length];
        for (int i = 0; i < length; i++)
        {
            window[i] = (float)Math.Pow(0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1))), power);
        }
        return window;
    }

    public static (float[,,] result, int frames) STFT(float[] audio, int n_fft, int hop_length, int win_length, float[] window)
    {
        if (audio == null || audio.Length == 0)
        {
            throw new ArgumentException("Audio signal cannot be null or empty.");
        }
        if (window.Length != win_length)
        {
            throw new ArgumentException("Window length must match win_length parameter.");
        }
        int numFreqBins = n_fft / 2 + 1;
        int numFrames = (int)Math.Ceiling((double)(audio.Length - win_length) / hop_length) + 1;
        if (audio.Length < win_length)
        {
            numFrames = 1;
        }
        float[,,] spectrogram = new float[numFreqBins, numFrames, 2];

        Complex32[] fftBuffer = new Complex32[n_fft];

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
        {
            int startIdx = frameIdx * hop_length;
            Array.Clear(fftBuffer, 0, n_fft);
            for (int i = 0; i < win_length; i++)
            {
                if (startIdx + i < audio.Length)
                {
                    fftBuffer[i] = new Complex32(audio[startIdx + i] * window[i], 0);
                }
                else
                {
                    fftBuffer[i] = new Complex32(0, 0);
                }
            }
            Fourier.Forward(fftBuffer, FourierOptions.Default);

            for (int k = 0; k < numFreqBins; k++)
            {
                spectrogram[k, frameIdx, 0] = fftBuffer[k].Real;
                spectrogram[k, frameIdx, 1] = fftBuffer[k].Imaginary;
            }
        }
        return (spectrogram, numFrames);
    }

    public static float[] ISTFT(float[,,] stft, int n_fft, int hop_length, int win_length, float[] window)
    {
        int numFrames = stft.GetLength(1);
        int numFreqBins = stft.GetLength(0);
        int outputLength = (numFrames - 1) * hop_length + n_fft;
        var outputSignal = new float[outputLength];
        float[] windowSum = new float[outputLength];
        Complex32[] fftBuffer = new Complex32[n_fft];
        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
        {
            Array.Clear(fftBuffer, 0, n_fft);
            for (int j = 0; j < numFreqBins; j++)
            {
                fftBuffer[j] = new Complex32(stft[j, frameIdx, 0], stft[j, frameIdx, 1]);
                if (j > 0 && j < n_fft / 2)
                {
                    fftBuffer[n_fft - j] = fftBuffer[j].Conjugate();
                }
            }
            if (n_fft % 2 == 0 && numFreqBins > n_fft / 2)
            {
                fftBuffer[n_fft / 2] = new Complex32(stft[n_fft / 2, frameIdx, 0], stft[n_fft / 2, frameIdx, 1]);
            }
            Fourier.Inverse(fftBuffer, FourierOptions.Default);
            int startIdx = frameIdx * hop_length;
            for (int j = 0; j < n_fft; j++)
            {
                if (startIdx + j < outputLength)
                {
                    outputSignal[startIdx + j] += fftBuffer[j].Real * window[j];
                    windowSum[startIdx + j] += window[j] * window[j];
                }
            }
        }
        for (int i = 0; i < outputLength; i++)
        {
            if (windowSum[i] > 1e-6)
            {
                outputSignal[i] /= windowSum[i];
            }
        }
        return outputSignal;
    }
}