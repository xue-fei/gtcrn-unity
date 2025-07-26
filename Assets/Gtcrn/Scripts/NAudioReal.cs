using NAudio.Wave;
using System;
using UnityEngine;

public class NAudioReal : MonoBehaviour
{
    WaveFormat format;
    WaveInEvent recorder;
    WaveOutEvent output;
    BufferedWaveProvider provider;
    string modelPath;
    GtcrnStreamNew gtcrn;

    // Start is called before the first frame update
    void Start()
    {
        format = new WaveFormat(16000, 1);
        recorder = new WaveInEvent()
        {
            WaveFormat = format,
            BufferMilliseconds = 16
        };
        output = new WaveOutEvent()
        {
            DesiredLatency = 100
        };
        provider = new BufferedWaveProvider(format)
        {
            DiscardOnBufferOverflow = true,
            BufferDuration = TimeSpan.FromMilliseconds(200)
        };
        output.Init(provider);

        recorder.DataAvailable += OnData;

        modelPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        gtcrn = new GtcrnStreamNew(modelPath);

        output.Play();
        recorder.StartRecording();
    }

    int count;
    float[] temp = new float[256];
    float[] enhancedOutput = new float[256];
    byte[] byts;
    // 频率16ms
    void OnData(object sender, WaveInEventArgs e)
    {
        try
        {
            temp = Util.BytesToFloat(e.Buffer);
            count = gtcrn.ProcessAudio(temp, temp.Length, out enhancedOutput);
            if (count > 0)
            {
                byts = Util.FloatToByte16(enhancedOutput);
                provider.AddSamples(byts, 0, byts.Length);
            }
        }
        catch (Exception ex)
        {
            Debug.Log(ex);
        }
    }

    void OnDestroy()
    {
        output.Stop();
        recorder.StopRecording();
    }
}