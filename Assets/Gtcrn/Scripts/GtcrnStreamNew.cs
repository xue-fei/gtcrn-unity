using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using UnityEngine;


public class GtcrnStreamNew : IDisposable
{
    // 常量与C代码严格一致
    private const int N_FFT = 512;
    private const int HOP_LENGTH = 256;
    private const int WIN_LENGTH = 512;
    private const int NUM_BINS = N_FFT / 2 + 1; // 257

    private readonly InferenceSession _onnxSession;

    // ONNX缓存张量（流式状态）
    private DenseTensor<float> _convCache;
    private DenseTensor<float> _traCache;
    private DenseTensor<float> _interCache;
    private readonly List<NamedOnnxValue> _inputTensors;
    private readonly DenseTensor<float> _inputMixTensor;

    // 音频缓冲区（模仿C的union结构）
    private struct AudioBuffer
    {
        public float[] All; // 总长度N_FFT=512
        public ArraySegment<float> Overlap; // 前HOP_LENGTH=256（重叠保留区）
        public ArraySegment<float> Input;   // 后HOP_LENGTH=256（新数据缓存区）
    }
    private readonly AudioBuffer _audioBuffer;
    private int _inputBufferSize; // Input区有效数据量

    // ISTFT重叠相加状态（对应C的out_frames_hop）
    private readonly float[] _outFramesHop;

    // STFT/ISTFT组件
    private readonly float[] _window;
    private readonly float[] _windowDivNfft; // window / N_FFT（与C的window_div_nff一致）
    private readonly Complex[] _fftBuffer;
    private readonly Complex[] _ifftBuffer;

    public GtcrnStreamNew(string modelPath)
    {
        var sessionOptions = new SessionOptions
        {
            // 改为1或与物理核心数一致（避免超线程带来的开销）
            InterOpNumThreads = 1,
            IntraOpNumThreads = Mathf.Max(1, SystemInfo.processorCount), // 直接使用核心数而非一半
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
        };
        // 初始化ONNX会话 
        _onnxSession = new InferenceSession(modelPath);

        // 初始化ONNX缓存（与C的缓存形状一致）
        _convCache = new DenseTensor<float>(dimensions: new[] { 2, 1, 16, 16, 33 });
        _traCache = new DenseTensor<float>(dimensions: new[] { 2, 3, 1, 1, 16 });
        _interCache = new DenseTensor<float>(dimensions: new[] { 2, 1, 33, 16 });
        _convCache.Fill(0);
        _traCache.Fill(0);
        _interCache.Fill(0);

        // 初始化输入张量列表（动态更新缓存引用）
        _inputMixTensor = new DenseTensor<float>(dimensions: new[] { 1, NUM_BINS, 1, 2 });
        _inputTensors = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("mix", _inputMixTensor),
            NamedOnnxValue.CreateFromTensor("conv_cache", _convCache),
            NamedOnnxValue.CreateFromTensor("tra_cache", _traCache),
            NamedOnnxValue.CreateFromTensor("inter_cache", _interCache)
        };

        // 音频缓冲区（模仿C的union，共享内存）
        _audioBuffer.All = new float[N_FFT];
        _audioBuffer.Overlap = new ArraySegment<float>(_audioBuffer.All, 0, HOP_LENGTH);
        _audioBuffer.Input = new ArraySegment<float>(_audioBuffer.All, HOP_LENGTH, HOP_LENGTH);
        _inputBufferSize = 0;

        // 重叠相加状态缓存
        _outFramesHop = new float[HOP_LENGTH];

        // 窗函数（与C一致：window_div_nff = window / N_FFT）
        _window = CreateHannWindow(WIN_LENGTH);
        _windowDivNfft = new float[WIN_LENGTH];
        for (int i = 0; i < WIN_LENGTH; i++)
            _windowDivNfft[i] = _window[i] / N_FFT;

        // FFT/IFFT缓冲区
        _fftBuffer = new Complex[N_FFT];
        _ifftBuffer = new Complex[N_FFT];

        UnityEngine.Debug.Log("初始化完成");
    }

    float[] stftInput = new float[N_FFT];
    Complex[] enhSpectrum = new Complex[NUM_BINS];
    Complex[] stftResult;
    float[] istftOutput;

    /// <summary>
    /// 处理流式音频数据（模仿C的gtcrn_stream_process_audio）
    /// </summary>
    /// <param name="inputAudio">输入PCM数据（16kHz，单声道，归一化）</param>
    /// <param name="numSamples">输入样本数</param>
    /// <param name="outputSamples">输出增强后的PCM数据（需外部释放）</param>
    /// <returns>输出样本数</returns>
    public int ProcessAudio(float[] inputAudio, int numSamples, out float[] outputSamples)
    {
        outputSamples = Array.Empty<float>();
        // 输入不足一帧（256样本）时缓存
        if (numSamples + _inputBufferSize < HOP_LENGTH)
        {
            Array.Copy(inputAudio, 0,
                      _audioBuffer.Input.Array, _audioBuffer.Input.Offset + _inputBufferSize,
                      numSamples);
            _inputBufferSize += numSamples;
            return 0;
        }

        // 计算可处理的帧数
        int totalAvailable = numSamples + _inputBufferSize;
        int numFrames = totalAvailable / HOP_LENGTH;
        int outputSize = numFrames * HOP_LENGTH;
        outputSamples = new float[outputSize];

        // 复制上一次的重叠数据到输出（C的out_frames_hop）
        Array.Copy(_outFramesHop, outputSamples, HOP_LENGTH);

        int inputOffset = 0; // 输入数据处理偏移量

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
        {
            // 1. 填充输入缓冲区（补充至256样本）
            int need = HOP_LENGTH - _inputBufferSize;
            if (need > 0)
            {
                int copy = Math.Min(need, numSamples - inputOffset);
                Array.Copy(inputAudio, inputOffset,
                          _audioBuffer.Input.Array, _audioBuffer.Input.Offset + _inputBufferSize,
                          copy);
                _inputBufferSize += copy;
                inputOffset += copy;
            }

            // 2. 准备STFT输入（overlap + input，共512样本）

            Array.Copy(_audioBuffer.Overlap.Array, _audioBuffer.Overlap.Offset, stftInput, 0, HOP_LENGTH);
            Array.Copy(_audioBuffer.Input.Array, _audioBuffer.Input.Offset, stftInput, HOP_LENGTH, HOP_LENGTH);

            // 应用窗函数
            for (int i = 0; i < WIN_LENGTH; i++)
            {
                stftInput[i] *= _window[i];
            }
            // 3. 执行STFT（模仿C的kiss_fftr）
            stftResult = STFT(stftInput);

            // 4. 填充ONNX输入张量（实部+虚部）
            for (int bin = 0; bin < NUM_BINS; bin++)
            {
                _inputMixTensor[0, bin, 0, 0] = (float)stftResult[bin].Real;
                _inputMixTensor[0, bin, 0, 1] = (float)stftResult[bin].Imaginary;
            }

            // 5. 执行ONNX推理（更新缓存引用）
            _inputTensors[1] = NamedOnnxValue.CreateFromTensor("conv_cache", _convCache);
            _inputTensors[2] = NamedOnnxValue.CreateFromTensor("tra_cache", _traCache);
            _inputTensors[3] = NamedOnnxValue.CreateFromTensor("inter_cache", _interCache);
            using var results = _onnxSession.Run(_inputTensors);

            // 6. 提取增强结果和更新缓存
            var enhTensor = results.First(r => r.Name == "enh").AsTensor<float>();
            _convCache = results.First(r => r.Name == "conv_cache_out").AsTensor<float>().ToDenseTensor();
            _traCache = results.First(r => r.Name == "tra_cache_out").AsTensor<float>().ToDenseTensor();
            _interCache = results.First(r => r.Name == "inter_cache_out").AsTensor<float>().ToDenseTensor();

            // 7. 转换增强结果为复数频谱

            for (int bin = 0; bin < NUM_BINS; bin++)
            {
                enhSpectrum[bin] = new Complex(enhTensor[0, bin, 0, 0], enhTensor[0, bin, 0, 1]);
            }
            // 8. 执行ISTFT（模仿C的kiss_fftri）
            istftOutput = ISTFT(enhSpectrum);

            // 9. 重叠相加（严格对齐C逻辑）
            // 前256样本叠加到输出
            for (int i = 0; i < HOP_LENGTH; i++)
            {
                outputSamples[frameIdx * HOP_LENGTH + i] += istftOutput[i] * _windowDivNfft[i];
            }
            // 非最后一帧：叠加256-512样本到输出
            if (frameIdx < numFrames - 1)
            {
                for (int i = HOP_LENGTH; i < WIN_LENGTH; i++)
                {
                    int outputPos = frameIdx * HOP_LENGTH + i;
                    if (outputPos < outputSize)
                    {
                        outputSamples[outputPos] += istftOutput[i] * _windowDivNfft[i];
                    }
                }
            }
            // 最后一帧：保存256-512样本作为下次overlap
            else
            {
                for (int i = 0; i < HOP_LENGTH; i++)
                {
                    _outFramesHop[i] = istftOutput[HOP_LENGTH + i] * _windowDivNfft[HOP_LENGTH + i];
                }
            }

            // 10. 更新overlap（将当前input移至overlap）
            Array.Copy(_audioBuffer.Input.Array, _audioBuffer.Input.Offset,
                      _audioBuffer.Overlap.Array, _audioBuffer.Overlap.Offset,
                      HOP_LENGTH);
            _inputBufferSize = 0;
        }

        // 缓存剩余输入数据
        int remaining = numSamples - inputOffset;
        if (remaining > 0)
        {
            Array.Copy(inputAudio, inputOffset,
                      _audioBuffer.Input.Array, _audioBuffer.Input.Offset + _inputBufferSize,
                      remaining);
            _inputBufferSize += remaining;
        }

        return outputSize;
    }

    Complex[] result = new Complex[NUM_BINS];
    /// <summary>
    /// 执行STFT（模仿C的kiss_fftr）
    /// </summary>
    private Complex[] STFT(float[] input)
    {
        // 填充FFT缓冲区（实部输入）
        for (int i = 0; i < N_FFT; i++)
        {
            _fftBuffer[i] = new Complex(input[i], 0);
        }
        // 执行FFT
        FFT(_fftBuffer, N_FFT);

        // 提取正频率分量（257个频段）

        Array.Copy(_fftBuffer, result, NUM_BINS);
        return result;
    }

    float[] result2 = new float[N_FFT];
    /// <summary>
    /// 执行ISTFT（模仿C的kiss_fftri）
    /// </summary>
    private float[] ISTFT(Complex[] spectrum)
    {
        // 填充完整频谱（含负频率共轭）
        Array.Copy(spectrum, _ifftBuffer, NUM_BINS);
        for (int i = 1; i < NUM_BINS - 1; i++)
        {
            _ifftBuffer[N_FFT - i] = Complex.Conjugate(_ifftBuffer[i]);
        }
        // 奈奎斯特分量强制为实数（C的kiss_fftri要求）
        if (N_FFT % 2 == 0)
        {
            _ifftBuffer[N_FFT / 2] = new Complex(_ifftBuffer[N_FFT / 2].Real, 0);
        }
        // 执行IFFT
        IFFT(_ifftBuffer, N_FFT);

        // 提取实部（C的kiss_fftri输出为实部）

        for (int i = 0; i < N_FFT; i++)
        {
            result2[i] = (float)_ifftBuffer[i].Real;
        }
        return result2;
    }

    /// <summary>
    /// 创建Hann窗（带平方根，与C一致）
    /// </summary>
    private float[] CreateHannWindow(int length)
    {
        float[] window = new float[length];
        for (int i = 0; i < length; i++)
        {
            window[i] = (float)Math.Sqrt(0.5f * (1 - Math.Cos(2 * Math.PI * i / (length - 1))));
        }
        return window;
    }

    /// <summary>
    /// FFT计算（模仿C的kiss_fft）
    /// </summary>
    private void FFT(Complex[] buffer, int length)
    {
        Fourier.Forward(buffer, FourierOptions.Matlab); // 与C的kiss_fft对齐
    }

    /// <summary>
    /// IFFT计算（模仿C的kiss_fft逆变换）
    /// </summary>
    private void IFFT(Complex[] buffer, int length)
    {
        // 共轭后FFT等价于IFFT（未缩放，与C一致）
        for (int i = 0; i < length; i++)
        {
            buffer[i] = Complex.Conjugate(buffer[i]);
        }
        FFT(buffer, length);

        for (int i = 0; i < length; i++)
        {
            buffer[i] = Complex.Conjugate(buffer[i]);
        }
    }

    public void Dispose()
    {
        _onnxSession?.Dispose();
        GC.SuppressFinalize(this);
    }
}