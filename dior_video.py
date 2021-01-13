import liankanstudio as lks
from liankanstudio.util.yt_util import parse_url, download, find_scenes_static
import cv2
import os
#First we download the video
video_path, caption_path = download(parse_url("https://www.youtube.com/watch?v=h4s0llOpKrU"), output_dir="../temp_dl")

# We open the capture
capture = lks.Capture(video_path)

# We find the scenes
capture.find_scenes()
#print(capture.scenes)
# We create a detector for person
#detector_custom = lks.Detector("torchfrcnn")
#cv2.imshow("r",capture.get_frame(29))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# Now we detect and track
#capture.detect_and_track(detector=detector_custom, rate=10, interactive=False, output_path="../temp_dl/test.avi")
#"torchfrcnn", "torchmrcnn","torchkrcnn","SSD" 
# 
threshold = {"yolo": 0.25, "torchfrcnn":0.25, "torchmrcnn":0.25,"torchkrcnn":0.25, "ssd":0.25}
for method in ["yolo", "torchfrcnn", "torchmrcnn","torchkrcnn", "ssd"]:
    print(method)
    detector_custom = lks.Detector(method)
    _ = capture.compute_stat(detector_custom, rate=500, verbose=True, output_path=os.path.join('temp_dl',f'{method}.evia'), identifier=None,
                            threshold_detect=threshold[method], merge_thresh=0.3, cpu_count=None, look_result=False, video_path=os.path.join('temp_dl',f'{method}.avi'))