using System.Collections.Generic;
using uMicrophoneWebGL;
using UnityEngine;

public class RealTest : MonoBehaviour
{
    public AudioPlayer player;
    public MicrophoneWebGL microphoneWebGL;
    GtcrnStream gtcrn;
    string modelPath;
    Queue<float> audioData = new Queue<float>();
    List<float> orginData = new List<float>();
    List<float> enhData = new List<float>();

    // Start is called before the first frame update
    void Start()
    {
        modelPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        gtcrn = new GtcrnStream(modelPath);
        microphoneWebGL.dataEvent.AddListener(OnData);
    }

    // Update is called once per frame
    void Update()
    {

    }

    /// <summary>
    /// https://github.com/Xiaobin-Rong/gtcrn/issues/47
    /// </summary>
    const int block = 512;
    float[] temp = new float[block];

    private void FixedUpdate()
    {
        if (audioData.Count > block)
        {
            for (int i = 0; i < block; i++)
            {
                temp[i] = audioData.Dequeue();
            }
            orginData.AddRange(temp);
            float[] enh = gtcrn.ProcessFrame(temp);
            player.AddData(enh);
            enhData.AddRange(enh);
        }
    }

    void OnData(float[] data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            audioData.Enqueue(data[i]);
        }
    }

    private void OnDestroy()
    {
        Util.SaveClip(1, 16000, orginData.ToArray(),
            Application.dataPath + "/zorgin.wav");
        Util.SaveClip(1, 16000, enhData.ToArray(),
            Application.dataPath + "/zenh.wav");
    }
}