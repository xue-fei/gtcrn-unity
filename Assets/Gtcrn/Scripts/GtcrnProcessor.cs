using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

public class GtcrnProcessor : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Stftprocessor _stftProcessor;

    private DenseTensor<float> _convCache;
    private DenseTensor<float> _traCache;
    private DenseTensor<float> _interCache;

    public GtcrnProcessor(string modelPath)
    {
        // 初始化ONNX Runtime会话
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
        };
        _session = new InferenceSession(modelPath, options);

        // 初始化STFT处理器
        _stftProcessor = new Stftprocessor();

        // 初始化缓存
        ResetState();
    }

    public void ResetState()
    {
        // 释放现有缓存
        _convCache = null;
        _traCache = null;
        _interCache = null;

        // 创建新的零初始化缓存
        _convCache = new DenseTensor<float>(new[] { 2, 1, 16, 16, 33 });
        _traCache = new DenseTensor<float>(new[] { 2, 3, 1, 1, 16 });
        _interCache = new DenseTensor<float>(new[] { 2, 1, 33, 16 });

        // 重置STFT状态
        _stftProcessor.Reset();
    }

    public float[] ProcessFrame(float[] audioFrame)
    {
        // 1. 计算STFT
        var stft = _stftProcessor.ComputeSTFT(audioFrame);

        // 2. 准备ONNX输入
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("mix", stft),
            NamedOnnxValue.CreateFromTensor("conv_cache", _convCache),
            NamedOnnxValue.CreateFromTensor("tra_cache", _traCache),
            NamedOnnxValue.CreateFromTensor("inter_cache", _interCache)
        };

        // 3. 运行推理
        using var outputs = _session.Run(inputs);

        // 4. 获取输出张量
        var enhanced = outputs.First().AsTensor<float>();
        _convCache = (DenseTensor<float>)outputs.First(o => o.Name == "conv_cache_out").AsTensor<float>();
        _traCache = (DenseTensor<float>)outputs.First(o => o.Name == "tra_cache_out").AsTensor<float>();
        _interCache = (DenseTensor<float>)outputs.First(o => o.Name == "inter_cache_out").AsTensor<float>();

        // 5. 计算ISTFT
        return _stftProcessor.ComputeISTFT(enhanced);
    }

    public void Dispose()
    {
        _session?.Dispose();
        _convCache = null;
        _traCache = null;
        _interCache = null;
    }
}