using System;
using UnityEngine;

public class GtcrnTest : MonoBehaviour
{
    GtcrnStream gtcrn;
    string modelPath;
    const int paddingLength = 512 / 2;

    // Start is called before the first frame update
    void Start()
    {
        modelPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        float[] longAudio = Util.ReadWav(Application.dataPath + "/mix.wav");

        gtcrn = new GtcrnStream(modelPath);
        string resultPath = Application.dataPath + "/result.wav";
        Loom.RunAsync(() =>
        {
            float[] paddedAudio = new float[paddingLength + longAudio.Length];
            Array.Copy(longAudio, 0, paddedAudio, paddingLength, longAudio.Length);
            float[] enhancedPaddedAudio = gtcrn.ProcessFrame(paddedAudio);
            float[] finalEnhancedAudio = new float[enhancedPaddedAudio.Length - paddingLength];
            Array.Copy(enhancedPaddedAudio, paddingLength, finalEnhancedAudio, 0, finalEnhancedAudio.Length);
            Util.SaveClip(1, 16000, finalEnhancedAudio, resultPath);
        });
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnDestroy()
    {
        //gtcrn.ResetState();
        gtcrn = null;
    }
}