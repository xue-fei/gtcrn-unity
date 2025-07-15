using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;

public class Stftprocessor
{
    private readonly int _nfft;
    private readonly int _hopLength;
    private readonly double[] _window;
    private readonly int _windowSize;
    private readonly int _numBins;
    private float[] _previousFrame;

    public Stftprocessor(int nfft = 512, int hopLength = 256)
    {
        _nfft = nfft;
        _hopLength = hopLength;
        _windowSize = nfft;
        _numBins = nfft / 2 + 1;

        // 创建汉宁窗的平方根
        _window = Window.Hann(_windowSize).Select(w => Math.Sqrt(w)).ToArray();

        _previousFrame = new float[_windowSize - _hopLength];
    }

    public DenseTensor<float> ComputeSTFT(float[] frame)
    {
        // 确保输入帧长度正确 (256采样点)
        if (frame.Length != _hopLength)
            throw new ArgumentException($"Input frame must be {_hopLength} samples");

        // 创建完整帧: 前一部分来自上一帧，当前帧，后补零
        var fullFrame = new float[_windowSize];
        Array.Copy(_previousFrame, 0, fullFrame, 0, _previousFrame.Length);
        Array.Copy(frame, 0, fullFrame, _previousFrame.Length, frame.Length);

        // 保存当前帧的后半部分作为下一帧的前半部分
        _previousFrame = frame.Skip(frame.Length - _previousFrame.Length).ToArray();

        // 应用窗函数
        var windowedFrame = new Complex32[_windowSize];
        for (int i = 0; i < _windowSize; i++)
        {
            windowedFrame[i] = new Complex32(
                (float)(fullFrame[i] * _window[i]),
                0
            );
        }

        // 执行FFT
        Fourier.Forward(windowedFrame, FourierOptions.Default);

        // 创建输出张量 [1, 257, 1, 2] - (batch, freq, time, real/imag)
        var tensor = new DenseTensor<float>(new[] { 1, _numBins, 1, 2 });

        // 填充结果 (只取前257个频率点)
        for (int f = 0; f < _numBins; f++)
        {
            tensor[0, f, 0, 0] = windowedFrame[f].Real;
            tensor[0, f, 0, 1] = windowedFrame[f].Imaginary;
        }

        return tensor;
    }

    public float[] ComputeISTFT(Tensor<float> stftTensor)
    {
        // 输入张量形状 [1, 257, 1, 2]
        if (stftTensor.Dimensions.Length != 4 ||
            stftTensor.Dimensions[1] != _numBins ||
            stftTensor.Dimensions[3] != 2)
            throw new ArgumentException("Invalid STFT tensor shape");

        // 创建复数数组
        var complexFrame = new Complex32[_windowSize];

        // 填充前257个频率点
        for (int f = 0; f < _numBins; f++)
        {
            complexFrame[f] = new Complex32(
                stftTensor[0, f, 0, 0],
                stftTensor[0, f, 0, 1]
            );
        }

        // 填充对称部分 (共轭对称)
        for (int f = 1; f < _numBins - 1; f++)
        {
            complexFrame[_windowSize - f] = Complex32.Conjugate(complexFrame[f]);
        }

        // 执行逆FFT
        Fourier.Inverse(complexFrame, FourierOptions.Default);

        // 应用窗函数并转换为实数
        var outputFrame = new float[_windowSize];
        for (int i = 0; i < _windowSize; i++)
        {
            outputFrame[i] = (float)(complexFrame[i].Real * _window[i]);
        }

        // 重叠相加处理 (只取中间256个点)
        var result = new float[_hopLength];
        Array.Copy(outputFrame, _windowSize - _hopLength, result, 0, _hopLength);

        return result;
    }

    public void Reset()
    {
        Array.Clear(_previousFrame, 0, _previousFrame.Length);
    }
}