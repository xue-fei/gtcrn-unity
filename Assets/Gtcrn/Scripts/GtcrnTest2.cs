using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class GtcrnTest2 : MonoBehaviour
{
    string simpleModelPath = "your_model_name.onnx";
    string outputPath = "test_wavs/enh_onnx.wav"; // Path to save the enhanced audio.
    int sampleRate = 16000; // Audio sample rate.
    int n_fft = 512; // FFT window size.
    int hop_length = 256; // Hop length between consecutive frames.
    int win_length = 512; // Window length (usually equals n_fft for STFT).

    // Start is called before the first frame update
    void Start()
    {
        simpleModelPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        outputPath = Application.streamingAssetsPath + "/result.wav";

        float[] rawAudio = ReadWav(Application.streamingAssetsPath + "/mix.wav");
        var window = Window.Hann(win_length).Select(x => (float)Math.Sqrt(x)).ToArray();
        (float[,,] stftResult, int frames) = ComputeSTFT(rawAudio, n_fft, hop_length, win_length, window);
        // 3. 初始化ONNX推理会话
        var session = new InferenceSession(simpleModelPath);

        var convCache = new DenseTensor<float>(new[] { 2, 1, 16, 16, 33 });
        var traCache = new DenseTensor<float>(new[] { 2, 3, 1, 1, 16 });
        var interCache = new DenseTensor<float>(new[] { 2, 1, 33, 16 });

        var outputs = new List<float[,,]>();
        var times = new List<double>();

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
            var sw = System.Diagnostics.Stopwatch.StartNew();
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
            sw.Stop();
            times.Add(sw.Elapsed.TotalMilliseconds);
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

        SaveClip(1, 16000, enhancedAudio, outputPath);
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

    float[] ReadWav(string filePath)
    {
        using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fs))
        {
            // 读取WAV文件头
            string riff = new string(reader.ReadChars(4));    // "RIFF"
            int fileSize = reader.ReadInt32();                // 文件总大小-8
            string wave = new string(reader.ReadChars(4));    // "WAVE"
            string fmt = new string(reader.ReadChars(4));     // "fmt "
            int fmtSize = reader.ReadInt32();                 // fmt块大小（至少16）

            // 读取音频格式信息
            short audioFormat = reader.ReadInt16();           // 1=PCM
            short numChannels = reader.ReadInt16();           // 通道数
            int sampleRate = reader.ReadInt32();              // 采样率
            int byteRate = reader.ReadInt32();                // 字节率
            short blockAlign = reader.ReadInt16();            // 块对齐
            short bitsPerSample = reader.ReadInt16();         // 采样深度

            // 验证文件格式
            if (riff != "RIFF" || wave != "WAVE" || fmt != "fmt ")
                throw new Exception("无效的WAV文件头");

            // 跳过fmt块的额外信息（如果有）
            if (fmtSize > 16)
                reader.ReadBytes(fmtSize - 16);

            // 查找数据块
            string dataChunkId;
            do
            {
                dataChunkId = new string(reader.ReadChars(4));
                if (dataChunkId != "data")
                    reader.ReadBytes(reader.ReadInt32()); // 跳过非数据块
            } while (dataChunkId != "data");

            int dataSize = reader.ReadInt32(); // 数据块大小（字节）

            // 验证音频参数
            if (audioFormat != 1)
                throw new Exception("仅支持PCM格式");
            if (numChannels != 1)
                throw new Exception("仅支持单声道音频");
            if (sampleRate != 16000)
                throw new Exception("仅支持16kHz采样率");
            if (bitsPerSample != 16)
                throw new Exception("仅支持16位采样深度");

            // 读取PCM数据并转换为float
            int sampleCount = dataSize / 2; // 16位 = 2字节/样本
            float[] floatData = new float[sampleCount];

            for (int i = 0; i < sampleCount; i++)
            {
                // 小端序读取16位样本
                byte lowByte = reader.ReadByte();
                byte highByte = reader.ReadByte();
                short pcmValue = (short)((highByte << 8) | lowByte);

                // 将16位PCM值转换为[-1.0, 1.0]范围的float
                floatData[i] = pcmValue / 32768.0f;
            }

            return floatData;
        }
    }

    void SaveClip(int channels, int frequency, float[] data, string filePath)
    {
        using (FileStream fileStream = new FileStream(filePath, FileMode.Create))
        {
            using (BinaryWriter writer = new BinaryWriter(fileStream))
            {
                // 写入RIFF头部标识
                writer.Write("RIFF".ToCharArray());
                // 写入文件总长度（后续填充）
                writer.Write(0);
                writer.Write("WAVE".ToCharArray());
                // 写入fmt子块
                writer.Write("fmt ".ToCharArray());
                writer.Write(16); // PCM格式块长度
                writer.Write((short)1); // PCM编码类型
                writer.Write((short)channels);
                writer.Write(frequency);
                writer.Write(frequency * channels * 2); // 字节率
                writer.Write((short)(channels * 2)); // 块对齐
                writer.Write((short)16); // 位深度
                                         // 写入data子块
                writer.Write("data".ToCharArray());
                writer.Write(data.Length * 2); // 音频数据字节数
                                               // 写入PCM数据（float转为short）
                foreach (float sample in data)
                {
                    writer.Write((short)(sample * 32767*100));
                }
                // 返回填充文件总长度
                fileStream.Position = 4;
                writer.Write((int)(fileStream.Length - 8));
            }
        }
    }
}