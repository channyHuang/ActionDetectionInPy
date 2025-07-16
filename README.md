# ActionDetectionInPy
Action Detect Using yolo+bytetrack+stgcn

# Basic Algorithm
1. [yolov5](https://github.com/ultralytics/yolov5) / [yolov8](https://github.com/ultralytics/ultralytics.git)
2. [BYTETracker](https://github.com/FoundationVision/ByteTrack)
3. [STGCN](https://github.com/hazdzz/STGCN)

# depend 
```sh
pip3 install opencv-python
pip3 install seaborn
pip3 install numpy
pip3 install torch
pip3 install lap
pip3 install cython_bbox
```

# 版本
torch 2.6以上修改了加载模型时的`weights_only`参数默认值
```sh
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
```

# Result
<video src="result.mp4" width="100%" controls></video>

![res_frame](./res_frame.jpg)