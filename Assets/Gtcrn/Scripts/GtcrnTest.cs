using UnityEngine;

public class GtcrnTest : MonoBehaviour
{
    GtcrnStream gtcrn;
    string modelPath;

    // Start is called before the first frame update
    void Start()
    {
        modelPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        float[] audioData = Util.ReadWav(Application.dataPath + "/mix.wav");

        gtcrn = new GtcrnStream(modelPath);
        string resultPath = Application.dataPath + "/result.wav";
        Loom.RunAsync(() =>
        {
            float[] processedFrame = gtcrn.ProcessFrame(audioData);
            Util.SaveClip(1, 16000, processedFrame, resultPath);
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