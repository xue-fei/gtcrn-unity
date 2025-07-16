using System.Collections.Generic;
using uMicrophoneWebGL;
using UnityEngine;

public class Test2 : MonoBehaviour
{
    public AudioPlayer player;
    public MicrophoneWebGL microphoneWebGL;
    GtcrnStream gtcrn;
    string modelPath;
    Queue<float> audioData = new Queue<float>();

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

    const int block = 256;
    float[] temp = new float[block];

    private void FixedUpdate()
    {
        if (audioData.Count > block)
        {
            for (int i = 0; i < block; i++)
            {
                temp[i] = audioData.Dequeue();
            }
            player.AddData(gtcrn.ProcessFrame(temp));
        }
    }

    void OnData(float[] data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            audioData.Enqueue(data[i]);
        }
    }
}