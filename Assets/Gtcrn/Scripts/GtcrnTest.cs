using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class GtcrnTest : MonoBehaviour
{
    StreamGTCRNProcessor audioEnhancer;
    string modelPath;
    public AudioClip audioClip;
    float[] audioData;
    List<float> result = new List<float>();

    // Start is called before the first frame update
    void Start()
    {
        modelPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        audioData = new float[audioClip.samples * audioClip.channels];
        audioClip.GetData(audioData, 0);

        audioEnhancer = new StreamGTCRNProcessor(modelPath);

        int blockSize = 256;
        int totalBlocks = (audioData.Length + blockSize - 1) / blockSize;
        for (int blockIndex = 0; blockIndex < totalBlocks; blockIndex++)
        {
            float[] block = new float[blockSize];
            int startIndex = blockIndex * blockSize;
            int elementsToCopy = Math.Min(blockSize, audioData.Length - startIndex);
            if (elementsToCopy > 0)
            {
                Array.Copy(audioData, startIndex, block, 0, elementsToCopy);
            }
            float[] processedFrame = audioEnhancer.ProcessFrame(block);
            result.AddRange(processedFrame);
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnDestroy()
    {
        audioEnhancer.ResetState();
        audioEnhancer = null;
        SaveClip(1, 16000, result.ToArray(), Application.streamingAssetsPath + "/result.wav");
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
                    writer.Write((short)(sample * 32767));
                }
                // 返回填充文件总长度
                fileStream.Position = 4;
                writer.Write((int)(fileStream.Length - 8));
            }
        }
    }
}