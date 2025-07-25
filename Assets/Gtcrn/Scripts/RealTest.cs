using System.Collections.Generic;
using uMicrophoneWebGL;
using UnityEngine;

public class RealTest : MonoBehaviour
{
    public AudioPlayer player;
    public MicrophoneWebGL microphoneWebGL;
    GtcrnStreamNew gtcrn;
    string modelPath;
    List<float> orginData = new List<float>();
    List<float> enhData = new List<float>();
    public bool isEnh = true;

    // Start is called before the first frame update
    void Start()
    {
        Application.targetFrameRate = 60;
        modelPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        gtcrn = new GtcrnStreamNew(modelPath);
        microphoneWebGL.dataEvent.AddListener(OnData);
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnGUI()
    {
        isEnh = GUI.Toggle(new Rect(300, 300, 100, 100), isEnh, isEnh ? "降噪开" : "降噪关");
    }

    float[] enhancedOutput = new float[256];
    int count;
    void OnData(float[] data)
    {
        orginData.AddRange(data);
        if (isEnh)
        {
            count = gtcrn.ProcessAudio(data, data.Length, out enhancedOutput);
            if (count > 0)
            {
                player.AddData(enhancedOutput);
                enhData.AddRange(enhancedOutput);
            }
        }
        else
        {
            player.AddData(data);
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