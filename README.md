# liankanstudio

## Purpose

Simple tool to do simple object detection on a video.

## Installation

Running for example
```
git clone https://github.com/Ailianlian/liankanstudio.git 
cd liankanstudio
pip install -r requirments.txt
python setup.py install
``` should be enough.

## Usage

You can choose to use it via python package or via cli.

### Python package

Here is a example to download a video from youtube and do person detection in it

```python
import liankanstudio as lks
from lks.util.yt_util import download, parse_url

# download the video from url
video_path, caption_path = download(parse_url("https://www.youtube.com/watch?v=YbJOTdZBX1g"), resolution="140p", output_dir="../temp_dl")

# We open the capture
capture = lks.Capture(video_path)

# We find the scenes in the video
capture.find_scenes()

# We create a detector using fast Rcnn pretrained in torch to detect person
detector_custom = lks.Detector("torchfrcnn", target="person")

# Now we detect it and write the video
capture.detect_and_track(detector=detector_custom, rate=10, interactive=False, output_path="../temp_dl/example.avi", release_end=True)
```

### CLI

A cli is provided. You can

- download a video, via (youtube) url, (youtube) video_id into video_path (relative or absolute).
```bash
liankan --url=<url> dl <url> <video_path>
liankan --no-url --res=240p dl <video_id> <video_path>
```
- process a video, by default we use yolo, detect person and use csrt as tracker. We detect every 10 frames and verbose is open by default.
```bash
liankan detect --target=person --method=yolo --tracker=csrt --rate=10 --verbose/--no-verbose --cache/--no-cache --look/--no-look <input_path> <output_path> 
```

#### Accepted tracker_type

Exhaustive list of the tracker_type used in this tool. Information can be found [here](https://docs.opencv.org/3.4.12/d2/d0a/tutorial_introduction_to_tracker.html)
- csrt      :
- kcf       :
- boosting  : 
- mil       :
- tld       :
- medianflow:
- mosse     :

#### Accepted detect_method
Exhaustive list of the detect_method
- ssd       : Use pretrained model, weights coming from [here](https://github.com/chuanqi305/MobileNet-SSD)
- torchfrcnn: Use the frcnn pretrained model of pytorch, can be found [here](https://pytorch.org/docs/stable/torchvision/models.html#faster-r-cnn)
- haar      : Use the haar cascade direclty into opencv. Details can be found [here](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html#:~:text=Haar-cascade%20Detection%20in%20OpenCV%20.%20OpenCV%20provides%20a,detect%20faces%20and%20eyes%20in%20an%20image%20)
- maskrcnn  : Use the frcnn pretrained model of pytorch, can be found [here](https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn)
- yolo      : Use a yolo-ssp pretrained model, can be found [here](https://pjreddie.com/darknet/yolo/)
- human     : Ask human to select a ROI.
