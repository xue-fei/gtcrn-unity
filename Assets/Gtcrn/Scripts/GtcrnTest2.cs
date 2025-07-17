using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class GtcrnTest2 : MonoBehaviour
{
    string simpleModelPath = "your_model_name.onnx";
    string outputPath = "test_wavs/enh_onnx.wav";
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
        var window = Window.Hann(win_length).Select(x => (float)Math.Sqrt(x)).ToArray();
        (float[,,] stftResult, int frames) = ComputeSTFT(rawAudio, n_fft, hop_length, win_length, window);
        // 3. 初始化ONNX推理会话
        var session = new InferenceSession(simpleModelPath);

        var convCache = new DenseTensor<float>(new[] { 2, 1, 16, 16, 33 });
        var traCache = new DenseTensor<float>(new[] { 2, 3, 1, 1, 16 });
        var interCache = new DenseTensor<float>(new[] { 2, 1, 33, 16 });

        var outputs = new List<float[,,]>();

        for (int i = 0; i < frames; i++)
        {
            // 准备当前帧
            var input = new DenseTensor<float>(new[] { 1, 257, 1, 2 });
            for (int j = 0; j < 257; j++)
            {
                input[0, j, 0, 0] = stftResult[i, j, 0]; // 实部
                input[0, j, 0, 1] = stftResult[i, j, 1]; // 虚部
            }

            // 准备输入
            var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("mix", input),
                    NamedOnnxValue.CreateFromTensor("conv_cache", convCache),
                    NamedOnnxValue.CreateFromTensor("tra_cache", traCache),
                    NamedOnnxValue.CreateFromTensor("inter_cache", interCache)
                };

            // 推理
            using (var results = session.Run(inputs))
            {
                var enh = results.First(t => t.Name == "enh").AsTensor<float>();
                convCache = (DenseTensor<float>)results.First(t => t.Name == "conv_cache_out").AsTensor<float>();
                traCache = (DenseTensor<float>)results.First(t => t.Name == "tra_cache_out").AsTensor<float>();
                interCache = (DenseTensor<float>)results.First(t => t.Name == "inter_cache_out").AsTensor<float>();

                // 存储输出
                var frameOutput = new float[1, 257, 2];
                for (int j = 0; j < 257; j++)
                {
                    frameOutput[0, j, 0] = enh[0, j, 0, 0];
                    frameOutput[0, j, 1] = enh[0, j, 0, 1];
                }
                outputs.Add(frameOutput);
            }
        }

        // 6. 合并所有帧
        var allFrames = new float[frames, 257, 2];
        for (int i = 0; i < frames; i++)
        {
            for (int j = 0; j < 257; j++)
            {
                allFrames[i, j, 0] = outputs[i][0, j, 0];
                allFrames[i, j, 1] = outputs[i][0, j, 1];
            }
        }

        // 7. 计算ISTFT
        float[] enhancedAudio = ComputeISTFT(allFrames, n_fft, hop_length, win_length, window);
        // 转换过程代码可能有误，此处把音量放大100倍
        Util.SaveClip(1, 16000, enhancedAudio, outputPath, 100f);
    }

    // Update is called once per frame
    void Update()
    {

    }

    static (float[,,] result, int frames) ComputeSTFT(float[] audio, int n_fft, int hop, int win, float[] window)
    {
        int frames = (audio.Length - n_fft) / hop + 1;
        var stft = new float[frames, n_fft / 2 + 1, 2]; // [frame, freq, real/imag]

        for (int i = 0; i < frames; i++)
        {
            // 提取帧并加窗
            var frame = new float[n_fft];
            Array.Copy(audio, i * hop, frame, 0, Math.Min(n_fft, audio.Length - i * hop));
            for (int j = 0; j < n_fft; j++) frame[j] *= window[j];

            // 计算FFT (使用MathNet.Numerics)
            var complexFrame = new Complex32[n_fft];
            for (int j = 0; j < n_fft; j++)
            {
                complexFrame[j] = new Complex32(frame[j], 0);
            }
            Fourier.Forward(complexFrame, FourierOptions.Default);

            // 存储结果（仅保留一半）
            for (int j = 0; j <= n_fft / 2; j++)
            {
                stft[i, j, 0] = complexFrame[j].Real;
                stft[i, j, 1] = complexFrame[j].Imaginary;
            }
        }
        return (stft, frames);
    }

    static float[] ComputeISTFT(float[,,] stft, int n_fft, int hop, int win, float[] window)
    {
        int frames = stft.GetLength(0);
        int outputLength = (frames - 1) * hop + n_fft;
        var output = new float[outputLength];
        var scale = window.Select(w => w * w).Sum(); // 用于归一化

        for (int i = 0; i < frames; i++)
        {
            // 重建完整频谱
            var fullSpectrum = new Complex32[n_fft];
            for (int j = 0; j <= n_fft / 2; j++)
            {
                fullSpectrum[j] = new Complex32(stft[i, j, 0], stft[i, j, 1]);
                if (j > 0 && j < n_fft / 2)
                {
                    fullSpectrum[n_fft - j] = fullSpectrum[j].Conjugate();
                }
            }

            // 逆FFT
            Fourier.Inverse(fullSpectrum, FourierOptions.Default);

            // 加窗并重叠相加
            int pos = i * hop;
            for (int j = 0; j < n_fft; j++)
            {
                if (pos + j < output.Length)
                {
                    output[pos + j] += fullSpectrum[j].Real * window[j] / scale;
                }
            }
        }
        return output;
    } 
}