using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

public class GtcrnStream7201 : IDisposable
{
    private InferenceSession onnxSession;
    // STFT parameters
    private const int N_FFT = 512;
    private const int HOP_LENGTH = 256;
    private const int WIN_LENGTH = 512;
    private const int SAMPLE_RATE = 16000;
    private readonly int numBins = N_FFT / 2 + 1;  // 257个频点

    // 模型缓存状态
    private DenseTensor<float> convCache;
    private DenseTensor<float> traCache;
    private DenseTensor<float> interCache;

    // 音频处理缓冲区
    private float[] inputBuffer;
    private float[] outputBuffer;
    private float[] window;
    private int bufferPosition = 0;

    public GtcrnStream7201(string modelPath)
    {
        onnxSession = new InferenceSession(modelPath);
        // 初始化模型缓存
        convCache = new DenseTensor<float>(new[] { 2, 1, 16, 16, 33 });
        traCache = new DenseTensor<float>(new[] { 2, 3, 1, 1, 16 });
        interCache = new DenseTensor<float>(new[] { 2, 1, 33, 16 });


        // 初始化音频缓冲区
        inputBuffer = new float[WIN_LENGTH];
        outputBuffer = new float[WIN_LENGTH];

        ResetCaches();

        window = CreateHannWindow(WIN_LENGTH);

        UnityEngine.Debug.Log("GTCRN流式处理器初始化完成");
    }

    public void ResetCaches()
    {
        // 重置所有缓存为0
        convCache.Buffer.Span.Fill(0);
        traCache.Buffer.Span.Fill(0);
        interCache.Buffer.Span.Fill(0);
        Array.Clear(inputBuffer, 0, inputBuffer.Length);
        Array.Clear(outputBuffer, 0, outputBuffer.Length);
        bufferPosition = 0;

        UnityEngine.Debug.Log("重置模型缓存");
    }

    public float[] ProcessFrame(float[] audioFrame)
    {
        if (audioFrame.Length != HOP_LENGTH)
        {
            throw new ArgumentException($"音频帧长度必须为{HOP_LENGTH}，当前为{audioFrame.Length}");
        }
        // 更新输入缓冲区
        UpdateInputBuffer(audioFrame);

        // 计算当前帧STFT
        Complex[] stftFrame = ComputeSTFTFrame();

        // 准备ONNX输入
        var inputTensor = new DenseTensor<float>(new[] { 1, numBins, 1, 2 });
        for (int bin = 0; bin < numBins; bin++)
        {
            inputTensor[0, bin, 0, 0] = (float)stftFrame[bin].Real;
            inputTensor[0, bin, 0, 1] = (float)stftFrame[bin].Imaginary;
        }

        // 运行ONNX推理
        using (var outputs = RunModel(inputTensor))
        {
            // 获取增强后的频谱
            var enhancedTensor = outputs.First(o => o.Name == "enh").AsTensor<float>();
            var enhancedFrame = new Complex[numBins];
            for (int bin = 0; bin < numBins; bin++)
            {
                enhancedFrame[bin] = new Complex(
                    enhancedTensor[0, bin, 0, 0],
                    enhancedTensor[0, bin, 0, 1]
                );
            }

            // 处理增强后的音频帧
            return ProcessEnhancedFrame(enhancedFrame);
        }
    }

    private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunModel(DenseTensor<float> input)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("mix", input),
            NamedOnnxValue.CreateFromTensor("conv_cache", convCache),
            NamedOnnxValue.CreateFromTensor("tra_cache", traCache),
            NamedOnnxValue.CreateFromTensor("inter_cache", interCache)
        };

        // 运行模型并更新缓存
        var outputs = onnxSession.Run(inputs);

        // 更新缓存状态
        convCache = outputs.First(o => o.Name == "conv_cache_out").AsTensor<float>().ToDenseTensor();
        traCache = outputs.First(o => o.Name == "tra_cache_out").AsTensor<float>().ToDenseTensor();
        interCache = outputs.First(o => o.Name == "inter_cache_out").AsTensor<float>().ToDenseTensor();

        return outputs;
    }

    private void UpdateInputBuffer(float[] newFrame)
    {
        // 滑动窗口：移除旧的HOP_LENGTH样本，添加新帧
        if (bufferPosition + HOP_LENGTH > WIN_LENGTH)
        {
            Array.Copy(inputBuffer, HOP_LENGTH, inputBuffer, 0, WIN_LENGTH - HOP_LENGTH);
            bufferPosition = WIN_LENGTH - HOP_LENGTH;
        }

        Array.Copy(newFrame, 0, inputBuffer, bufferPosition, HOP_LENGTH);
        bufferPosition += HOP_LENGTH;
    }

    private Complex[] ComputeSTFTFrame()
    {
        // 加窗处理
        var windowedFrame = new float[WIN_LENGTH];
        for (int i = 0; i < WIN_LENGTH; i++)
        {
            windowedFrame[i] = inputBuffer[i] * window[i];
        }

        // 执行FFT
        var complexBuffer = new Complex[N_FFT];
        for (int i = 0; i < WIN_LENGTH; i++)
        {
            complexBuffer[i] = new Complex(windowedFrame[i], 0);
        }

        FFT(complexBuffer, N_FFT);

        // 返回正频率部分 (0 ~ Nyquist)
        return complexBuffer.Take(numBins).ToArray();
    }

    private float[] ProcessEnhancedFrame(Complex[] enhancedSTFT)
    {
        // 重建完整频谱 (共轭对称)
        var fullSpectrum = new Complex[N_FFT];
        for (int i = 0; i < numBins; i++)
        {
            fullSpectrum[i] = enhancedSTFT[i];
        }

        // 填充负频率 (共轭对称)
        for (int i = 1; i < numBins - 1; i++)
        {
            fullSpectrum[N_FFT - i] = Complex.Conjugate(enhancedSTFT[i]);
        }

        // 执行IFFT
        IFFT(fullSpectrum, N_FFT);

        // 加窗和重叠相加
        var outputFrame = new float[HOP_LENGTH];
        for (int i = 0; i < WIN_LENGTH; i++)
        {
            float sample = (float)fullSpectrum[i].Real * window[i];

            // 重叠相加
            if (i < HOP_LENGTH)
            {
                outputFrame[i] = sample + outputBuffer[i];
                outputBuffer[i] = 0; // 重置已使用的部分
            }
            else
            {
                outputBuffer[i - HOP_LENGTH] += sample;
            }
        }

        // 前移缓冲区
        Array.Copy(outputBuffer, HOP_LENGTH, outputBuffer, 0, WIN_LENGTH - HOP_LENGTH);
        Array.Clear(outputBuffer, WIN_LENGTH - HOP_LENGTH, HOP_LENGTH);

        return outputFrame;
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

    public void Dispose()
    {
        onnxSession?.Dispose();
    }
}