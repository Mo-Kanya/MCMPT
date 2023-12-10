# Multi-Camera Mutli-People Tracking
Real-time tracking-by-detection pipeline using MVDeTr and DeepSORT

![](sample_outputs/sample_outputs.gif)

For detailed info, refer to our [website](https://mscvprojects.ri.cmu.edu/f23team7/).

## Installation
- Python 3.9
- PyTorch 2.0.0
- GPU required

All dependencies are listed in `requirements.txt`.

## Execution
```
python3 main.py --dataset_config_file [dataset config file path]
                --outdir [output directory path]
```

NOTE: Sample outputs can be found in `sample_outputs`.

## File Structures
```
.
├── dataset_config
├── deep_sort                       # DeepSORT tracker
├── mmp_tracking_helper
├── model_weights
│   ├── MultiviewDetector.pth       # MVDeTr
│   ├── osnet_x0_25_market1501.pt   # appearance encoder for DeepSORT tracker
│   └── yolo_finetuned.pt           # YOLOv8 finetuned on MMPTrack dataset
├── multiview_detector              # MVDeTr multi-view detector
├── reid                            # appearance descriptor encoder for tracker
├── sample_outputs
├── main.py
├── psort.py                        # SORT using point instead of bounding box
├── requirements.txt
├── track_psort.py                  # tracking demo using point-SORT
└── README.md
```