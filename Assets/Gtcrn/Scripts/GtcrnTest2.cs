using UnityEngine;

public class GtcrnTest2 : MonoBehaviour
{
    GtcrnStreamNew gtcrn;
    string modelPath;

    // Start is called before the first frame update
    void Start()
    {
        modelPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        float[] longAudio = Util.ReadWav(Application.dataPath + "/mix.wav");

        gtcrn = new GtcrnStreamNew(modelPath);
        string resultPath = Application.dataPath + "/result.wav";
        Loom.RunAsync(() =>
        {
            float[] enhancedAudio = new float[longAudio.Length];
            int count = gtcrn.ProcessAudio(longAudio, longAudio.Length, out enhancedAudio);
            if (count > 0)
            {
                Util.SaveClip(1, 16000, enhancedAudio, resultPath);
            }
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