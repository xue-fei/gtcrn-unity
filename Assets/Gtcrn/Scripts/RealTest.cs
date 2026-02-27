using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using uMicrophoneWebGL;
using UnityEngine;
using UnityEngine.Networking;

public class RealTest : MonoBehaviour
{
    public AudioPlayer player;
    public MicrophoneWebGL microphoneWebGL;
    GtcrnStreamNew gtcrn;
    string localPath;
    string sbPath;
    List<float> orginData = new List<float>();
    List<float> enhData = new List<float>();
    public bool isEnh = true;

    // Start is called before the first frame update
    void Start()
    {
        Application.targetFrameRate = 60;

        localPath = Application.streamingAssetsPath + "/gtcrn_simple.onnx";
        sbPath = Application.persistentDataPath + "/gtcrn_simple.onnx";

        if (Application.platform == RuntimePlatform.Android
            || Application.platform == RuntimePlatform.IPhonePlayer)
        {
            if (!File.Exists(sbPath))
            {
                StartCoroutine(CopyModel(localPath, Application.persistentDataPath));
            }
            else
            {
                gtcrn = new GtcrnStreamNew(sbPath);
                microphoneWebGL.dataEvent.AddListener(OnData);
            }
        }
        else
        {
            gtcrn = new GtcrnStreamNew(localPath);
            microphoneWebGL.dataEvent.AddListener(OnData);
        }
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

    IEnumerator CopyModel(string sourcePath, string destPath)
    {
        using (UnityWebRequest www = UnityWebRequest.Get(sourcePath))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    // 写入文件
                    File.WriteAllBytes(sbPath, www.downloadHandler.data);
                    Debug.Log($"复制成功：{sbPath}");

                    gtcrn = new GtcrnStreamNew(sbPath);
                    microphoneWebGL.dataEvent.AddListener(OnData);
                }
                catch (Exception e)
                {
                    Debug.LogError($"写入失败：{e.Message}");
                }
            }
            else
            {
                Debug.LogError($"读取失败：{www.error}");
            }
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