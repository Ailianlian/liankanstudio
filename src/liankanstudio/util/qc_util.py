"""
File to welcome all util function useful for qc
"""
import cv2
import numpy as np
#(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#CV2_MAJOR_VER = int(major_ver)
#CV2_MINOR_VER = int(minor_ver)
def qc_scene(capture, list_scenes, fps=100):
    """
    Show the video with at a given framerate with scene id written in corner.

    Args
    ----

    capture     : Capture object, simple wrapper around cv2 videoCapture object. Possible it disappear when better understanding of videocapture object.
    list_scenes : List of scenes, to compare it in a single qc.
    fps         : The fps speed we want to have during the qc.
    """
    list_processed_scenes = np.zeros((len(list_scenes),list_scenes[-1][-1][1].get_frames()))
    for k,scenes in enumerate(list_scenes):
        for a,b in scenes:
            list_processed_scenes[k][a.get_frames():]+=1
    def _scene_text(img, frame_num):
        for k in range(len(list_scenes)):
            cv2.putText(img, f"Scene numero {list_processed_scenes[k][frame_num]}.", (100,80+k*50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0),2)
        return img, None
    
    capture.custom_process(func=_scene_text, fps=fps, window_name="Scene Check")

def qc_box_img(img, boxs, colors=(255,0,0)):
    for bbox in boxs:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
    cv2.imshow("QC BOX",img)
    cv2.waitKey(0) & 0xff
    cv2.destroyAllWindows()
