using System;
using System.Linq;
using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;

public class GtcrnStream : IDisposable
{
    private InferenceSession onnxSession;
    // STFT 参数
    private const int N_FFT = 512;
    private const int HOP_LENGTH = 256;
    private const int WIN_LENGTH = 512;
    private const int SAMPLE_RATE = 16000;

    // ONNX 模型缓存，用于流式推理时保持状态
    private DenseTensor<float> convCache;
    private DenseTensor<float> traCache;
    private DenseTensor<float> interCache;

    public GtcrnStream(string modelPath)
    {
        onnxSession = new InferenceSession(modelPath);

        // 初始化 ONNX 模型的内部缓存，只执行一次
        // 这些缓存将用于在连续的 ProcessFrame 调用中维护模型状态
        convCache = new DenseTensor<float>(dimensions: new[] { 2, 1, 16, 16, 33 });
        traCache = new DenseTensor<float>(dimensions: new[] { 2, 3, 1, 1, 16 });
        interCache = new DenseTensor<float>(dimensions: new[] { 2, 1, 33, 16 });

        // 确保缓存初始值为零，与 Python 参考代码一致
        convCache.Fill(0.0f);
        traCache.Fill(0.0f);
        interCache.Fill(0.0f);

        UnityEngine.Debug.Log("初始化完成");
    }

    /// <summary>
    /// 处理单个音频帧以进行增强
    /// </summary>
    /// <param name="audioFrame">输入音频帧（例如 512 个浮点数）</param>
    /// <returns>增强后的音频帧</returns>
    public float[] ProcessFrame(float[] audioFrame)
    {
        // 对输入音频帧执行短时傅里叶变换 (STFT)
        // 对于 512 样本的 audioFrame，这将产生一个 STFT 帧
        Complex[,] stft = STFT(audioFrame);
        // 频段数 (N_FFT / 2 + 1)
        int numBins = stft.GetLength(0);
        // STFT 帧数 (对于 512 样本输入通常为 1)
        int numFrames = stft.GetLength(1);
        UnityEngine.Debug.Log("STFT 帧数: " + numFrames);
        // 增强后的 STFT 结果将存储在这里
        Complex[,] enhancedSTFT = new Complex[numBins, numFrames];
        // 遍历 STFT 帧（对于 512 样本输入，此循环只运行一次）
        for (int frame = 0; frame < numFrames; frame++)
        {
            // 为当前 STFT 帧准备 ONNX 输入张量
            // 形状: [批次大小, 频段数, 1, 复数分量 (实部/虚部)]
            var inputTensor = new DenseTensor<float>(dimensions: new[] { 1, numBins, 1, 2 });
            // 填充输入张量，将复数 STFT 值拆分为实部和虚部
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
            // 运行 ONNX 模型推理
            using var results = onnxSession.Run(inputs);

            // 从推理结果中提取增强后的帧和更新的缓存
            var enhancedFrame = results.First(r => r.Name == "enh").AsTensor<float>();
            convCache = results.First(r => r.Name == "conv_cache_out").AsTensor<float>().ToDenseTensor();
            traCache = results.First(r => r.Name == "tra_cache_out").AsTensor<float>().ToDenseTensor();
            interCache = results.First(r => r.Name == "inter_cache_out").AsTensor<float>().ToDenseTensor();

            // 将增强后的实部和虚部重新组合成复数，存储到 enhancedSTFT 数组中
            for (int bin = 0; bin < numBins; bin++)
            {
                enhancedSTFT[bin, frame] = new Complex(
                    enhancedFrame[0, bin, 0, 0],
                    enhancedFrame[0, bin, 0, 1]
                );
            }
        }
        // 对增强后的 STFT 执行逆短时傅里叶变换 (ISTFT) 以获取时域音频
        return ISTFT(enhancedSTFT);
    }

    /// <summary>
    /// 执行短时傅里叶变换 (STFT)
    /// </summary>
    /// <param name="audio">输入音频数据</param>
    /// <returns>复数形式的 STFT 结果</returns>
    public Complex[,] STFT(float[] audio)
    {
        // 创建 Hann 窗函数，包含平方根操作以匹配 Python 参考
        var windowFunc = CreateHannWindow(WIN_LENGTH);
        // 计算 STFT 帧数
        // 如果 audio.Length 小于 WIN_LENGTH，numFrames 可能会是 0 或负数
        // 确保至少有一个帧，即使需要零填充
        var numFrames = (audio.Length - WIN_LENGTH) / HOP_LENGTH + 1;
        if (numFrames <= 0)
        {
            numFrames = 1; // 确保至少处理一个帧，可能需要零填充
        }
        // 频段数
        var numBins = N_FFT / 2 + 1;

        var stftResult = new Complex[numBins, numFrames];
        // 用于 FFT 计算的缓冲区
        var fftBuffer = new Complex[N_FFT];

        for (int frame = 0; frame < numFrames; frame++)
        {
            // 当前帧的起始索引
            var frameStart = frame * HOP_LENGTH;

            // 应用窗函数并复制到 FFT 缓冲区
            for (int i = 0; i < WIN_LENGTH; i++)
            {
                if (frameStart + i < audio.Length)
                {
                    // 将音频样本乘以窗函数，并作为复数的实部
                    fftBuffer[i] = new Complex(audio[frameStart + i] * windowFunc[i], 0);
                }
                else
                {
                    // 如果帧超出音频长度，则进行零填充
                    fftBuffer[i] = Complex.Zero;
                }
            }

            // 如果 WIN_LENGTH 小于 N_FFT，则对 FFT 缓冲区的其余部分进行零填充
            for (int i = WIN_LENGTH; i < N_FFT; i++)
            {
                fftBuffer[i] = Complex.Zero;
            }

            // 执行快速傅里叶变换 (FFT)
            FFT(fftBuffer, N_FFT);

            // 复制正频率部分（频谱的“一半”）
            for (int bin = 0; bin < numBins; bin++)
            {
                stftResult[bin, frame] = fftBuffer[bin];
            }
        }

        return stftResult;
    }

    /// <summary>
    /// 创建 Hann 窗函数，并应用平方根（与 Python 参考一致）
    /// </summary>
    /// <param name="length">窗函数长度</param>
    /// <returns>Hann 窗函数数组</returns>
    private float[] CreateHannWindow(int length)
    {
        var window = new float[length];
        for (int i = 0; i < length; i++)
        {
            // 标准 Hann 窗函数公式
            window[i] = (float)(0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1))));
        }

        // 按照 Python 参考代码应用平方根
        for (int i = 0; i < length; i++)
        {
            window[i] = (float)Math.Sqrt(window[i]);
        }

        return window;
    }

    /// <summary>
    /// 快速傅里叶变换 (FFT)
    /// </summary>
    /// <param name="buffer">要进行 FFT 的复数缓冲区（会原地修改）</param>
    /// <param name="length">缓冲区长度（必须是 2 的幂）</param>
    private void FFT(Complex[] buffer, int length)
    {
        // 位反转置换
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
        // 蝶形运算
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
    /// 执行逆短时傅里叶变换 (ISTFT)
    /// </summary>
    /// <param name="stft">复数形式的 STFT 结果</param>
    /// <returns>时域音频数据</returns>
    public float[] ISTFT(Complex[,] stft)
    {
        var numBins = stft.GetLength(0);
        var numFrames = stft.GetLength(1);
        // 使用与 STFT 相同的窗函数
        var windowFunc = CreateHannWindow(WIN_LENGTH);
        // 计算输出音频的长度
        var outputLength = (numFrames - 1) * HOP_LENGTH + WIN_LENGTH;
        // 存储最终输出音频
        var output = new float[outputLength];
        // 用于重叠相加归一化
        var windowSum = new float[outputLength];
        // 用于 IFFT 计算的缓冲区
        var ifftBuffer = new Complex[N_FFT];

        for (int frame = 0; frame < numFrames; frame++)
        {
            // 准备 IFFT 输入：复制正频率部分
            for (int bin = 0; bin < numBins; bin++)
            {
                ifftBuffer[bin] = stft[bin, frame];
            }
            // 镜像负频率部分，利用实值信号的对称性
            // 排除直流分量 (bin 0) 和奈奎斯特分量 (numBins - 1)
            for (int bin = 1; bin < numBins - 1; bin++)
            {
                ifftBuffer[N_FFT - bin] = Complex.Conjugate(ifftBuffer[bin]);
            }
            // 对于偶数 N_FFT，奈奎斯特分量（如果存在）应该是纯实数
            if (N_FFT % 2 == 0 && numBins == N_FFT / 2 + 1)
            {
                ifftBuffer[N_FFT / 2] = new Complex(ifftBuffer[N_FFT / 2].Real, 0);
            }
            // 执行逆快速傅里叶变换 (IFFT)
            IFFT(ifftBuffer, N_FFT);

            // 重叠相加 (Overlap-Add)
            // 当前帧在输出中的起始位置
            var frameStart = frame * HOP_LENGTH;
            for (int i = 0; i < WIN_LENGTH; i++)
            {
                if (frameStart + i < outputLength)
                {
                    // 将 IFFT 结果的实部乘以窗函数，并累加到输出中
                    output[frameStart + i] += (float)ifftBuffer[i].Real * windowFunc[i];
                    // 累加窗函数的平方，用于后续的归一化
                    windowSum[frameStart + i] += windowFunc[i] * windowFunc[i];
                }
            }
        }

        // 归一化重叠相加的结果
        for (int i = 0; i < outputLength; i++)
        {
            if (windowSum[i] > 1e-10f)
            {
                output[i] /= windowSum[i];
            }
            else
            {
                output[i] = 0.0f; // 如果没有窗函数贡献，则设置为零
            }
        }

        return output;
    }

    /// <summary>
    /// 逆快速傅里叶变换 (IFFT)
    /// </summary>
    /// <param name="buffer">要进行 IFFT 的复数缓冲区（会原地修改）</param>
    /// <param name="length">缓冲区长度（必须是 2 的幂）</param>
    private void IFFT(Complex[] buffer, int length)
    {
        // 对输入进行共轭
        for (int i = 0; i < length; i++)
        {
            buffer[i] = Complex.Conjugate(buffer[i]);
        }

        // 执行 FFT
        FFT(buffer, length);

        // 对输出进行共轭并缩放
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