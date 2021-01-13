import cv2
import tqdm
from .util.yt_util import parse_url, download, find_scenes_static
from .util.mp_util import dill_map
from .tracking import TrackableObject
import numpy as np
import multiprocessing as mp
import _pickle as cp
import os
import copy

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
CV2_MAJOR_VER = int(major_ver)
CV2_MINOR_VER = int(minor_ver)

# TODO : Should do something like Capture(cv2.VideoCapture), would clean interface and avoid overhead of copy feature.

def _get_detection_frames(rate, scenes, max_frame, verbose, cut_scene=False):
    last_frame=0
    frames = [0]
    if scenes is not None:
        if cut_scene:
            frames = [[0]]
        max_frame = scenes[-1][1].get_frames()
        if verbose:
            to_iterate = tqdm.tqdm(range(max_frame))
        else:
            to_iterate = range(max_frame)
        scene_counter=1
        for frame_nb in to_iterate:
            #check we don't have a new scene
            if scene_counter<len(scenes) and frame_nb == scenes[scene_counter][0].get_frames():
                last_frame = frame_nb
                if cut_scene:
                    frames +=[[0]]
                else:
                    frames +=[frame_nb]
                scene_counter+=1
            elif frame_nb==last_frame+rate:
                last_frame=frame_nb
                if cut_scene:
                    frames[-1] +=[frame_nb-scenes[scene_counter-1][0].get_frames()]
                else:
                    frames +=[frame_nb]
    else:
        if verbose:
            to_iterate = tqdm.tqdm(range(max_frame))
        else:
            to_iterate = range(max_frame)
        for frame_nb in to_iterate:
            if frame_nb==last_frame+rate:
                last_frame=frame_nb
                frames+=[frame_nb]
    return frames

def _detect(img, detector, conf_thresh, nms_thresh, trackables=None, ret_all=False):
    boxes, conf, labels = detector.detect(img)
    # we clean boxs
    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), conf_thresh, nms_thresh)
    if len(idxs)>0:
        nboxes = boxes[idxs.flatten()]
    else:
        nboxes = []
    if ret_all:
        return nboxes, conf, labels, boxes
    return nboxes


def _identify(img, boxes, trackables, obj_counter, frame_nb, tracker_type):
    # for now we just erase all the old object
    return {obj_counter+k : TrackableObject(obj_counter+k,img=img, box=boxes[k], frame_nb=frame_nb, tracker_type=tracker_type) for k in range(len(boxes))}

def _track(img, trackables):
    tracks = list(trackables.keys())
    for tracked in tracks:
        if not trackables[tracked].update_box(img):
            del trackables[tracked]
    return trackables

class Capture(object):
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.major_ver = CV2_MAJOR_VER
        self.minor_ver = CV2_MINOR_VER
        self.is_valid = False
        self.frame_counter = 0
        self.scenes = None
        if self.cap.isOpened():
            ok, frame = self.cap.read()
            self.h, self.w = frame.shape[0:2]
            if ok:
                self._set_frame(0)
                self.is_valid = True
            # look https://stackoverflow.com/questions/10057234/opencv-how-to-restart-a-video-when-it-finishes for this
            if int(self.major_ver)>=4:
                self.max_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            else:
                self.max_frame = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        #self.release()
        self.by_scene = []
    
    def _set_frame(self, frame_nb):
        if int(self.major_ver)>=4:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
        else:
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_nb)
        self.frame_counter = frame_nb

    def get_frame(self, frame_nb):
        self._set_frame(frame_nb)
        ok, frame = self.get_next_frame()
        return frame

    def show_frame(self, frame_nb):
        """
        Output the given frame, simple utility
        """
        cv2.imshow(f"Frame {frame_nb}",self.get_frame(frame_nb))
        cv2.waitKey(0) & 0xff
        cv2.destroyAllWindows()

    def show_frames(self, frames):
        """
        Output the given frame, simple utility
        """
        for frame in frames:
            cv2.imshow(f"Frame {frame_nb}",self.get_frame(frame_nb))
            k = cv2.waitKey(0) & 0xff
            if k == 27 :
                break
        cv2.destroyAllWindows()

    def reset(self):
        self._set_frame(0)
    
    def get_next_frame(self, skip=0):
        cnt=skip
        ok=True
        while cnt>0 and ok:
            ok, frame = self.cap.read()
            if ok:
                self.frame_counter+=1
            # I have to see how to catch the end of the video
            #else:
            #    raise Exception(f"A problem occurs when reading the frame {self.frame_counter}")
        ok, frame = self.cap.read()
        if ok:
            self.frame_counter+=1
        #else:
        #    raise Exception(f"A problem occurs when reading the frame {self.frame_counter}")
        return ok, frame
    
    def release(self):
        self.cap.release()

    def custom_process(self, output_path=None, verbose=True, func=None,reset=True, release_end=False, window_name=None,
                        ctrl_func=None, dyn_ctrl_fps=False,fps=-1, max_frame=-1):
        """
        Read a video as fast as possible and process it according to the function

        Args
        ----

        output_path : the path where we write the video if not None
        verbose     : if true about nice information about the processing
        func        : Function of frame, frame_counter , return a frame. Apply to each frame
        fps         : The targeted fps if -1, we try to go as fast as possible.
        window_name : The name of the window created by opencv. If None we don't do video
        ctrl_func   : Callback function on a slider with range 1 -> 256
        dyn_ctrl_fps: True if you want to control dynamically the fps
        
        Note
        ----
        
        The framerate is not perfect, as many things in the package for the moment :D
        ctrl_func and dyn_ctrl_fps will disappear when the Studio object will be created
        Studio object will handle all the context around reading a video like?
        """
        if max_frame==-1:
            max_frame=self.max_frame+1
        if output_path is not None:
            print(output_path)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path,fourcc,25,(self.w,self.h))
        if reset:
            self.reset()
        if verbose:
            pbar = tqdm.tqdm(total=max_frame, unit=" frames ", smoothing=0.1)
        frame_counter = 0
        # max_frame can be false so we will simply use it as a guess
        isprocessing = True
        ms_time = 1/fps*1000
        ctx=None
        if window_name is not None:
            cv2.namedWindow(window_name)
            # we will show some thing
            if ctrl_func is not None or dyn_ctrl_fps:
                cv2.namedWindow('ctrl')
            if ctrl_func is not None:
                cv2.createTrackbar( 'user_func', 'ctrl', 128, 255, ctrl_func )
            if dyn_ctrl_fps:
                def filler(value):
                    pass
                cv2.createTrackbar( 'fps_ctrl', 'ctrl', max(fps,1), 255, filler )
        while isprocessing and frame_counter<max_frame:
            # Read a new frame
            ok, frame = self.get_next_frame()
            if not ok:
                # well should be because we finish the video?
                break
                isprocessing=False
            #We begin the time count, we use opencv and not time
            ticks = cv2.getTickCount()

            # apply the function
            if func is not None:
                if ctx is None:
                    frame, ctx = func(frame, frame_counter)
                else:
                    frame, ctx = func(frame, frame_counter, *ctx)
            if window_name:
                if ctrl_func is not None:
                    user_value = cv2.getTrackbarPos('user_func','ctrl')
                if dyn_ctrl_fps:
                    fps = cv2.getTrackbarPos('fps_ctrl','ctrl') + 1
                cv2.imshow(window_name, frame)
                #cv2.putText(frame, f"Current FPS : {fps}.", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0),2)
            ms_time = 1/max(fps,1)*1000
            
            # Update the frame counter, why we have this one since we have self.frame_counter?
            frame_counter += 1
            if output_path is not None:
                writer.write(frame)
            # Calculate time since last ticks
            frame_time_ms = (cv2.getTickCount() - ticks)/ cv2.getTickFrequency() * 1000
            # we allow user to quit before the end, since why not.
            if fps == -1:
                k = cv2.waitKey(1) & 0xFF
            else:
                if frame_time_ms<ms_time:
                    #we wait
                    k = cv2.waitKey(int(ms_time - frame_time_ms)) & 0xFF
                else:
                    # we already go slower...
                    k = cv2.waitKey(1) & 0xFF
            
            if k == ord("q") : break
            if frame_counter<=self.max_frame and verbose:
                pbar.update(1)
        if release_end:
            self.release()
        if verbose:
            pbar.close()
        if output_path is not None:
            writer.release()
        cv2.destroyAllWindows()

    def find_scenes(self, automatic=False, threshold=None):
        """
        Find scenes and populate the scenes attribute. can be also used outside of Capture via util.find_scene_static

        Args
        ----
        automatic   : analyse the statistics computed to get scenes in order to have the best cut.
        threshold   : if not None, it bypass automatic.

        """
        self.scenes = find_scenes_static(self.video_path, automatic=False, threshold=None)
        self.max_frame = self.scenes[-1][1].get_frames()

    def detect_and_track(self, detector, rate, scenes=None, tracker_type="csrt",  threshold_detect=0.8, merge_thresh=0.3,verbose=True, 
                        output_path=None, identifier=None, interactive=True, fps=30, max_frame=-1):
        """
        Use the tracker and detector given as parameter to create a new video.

        Args
        ----

        detector        : an object Detector. Will detect the element according to the setting of the object.
        rate            : if no scenes are found, do detection every rate frames.
        scenes          : the scenes , overwrite self.scenes if not NONE, if scene and self.scene are none we compute the scene
        tracker_type    : the different opencv implemented tracker, value in ["csrt","kcf","boosting","mil","tld","medianflow","mosse"]
        threshold_detect: threshold for the detection algorithm, under this level of confidence no box. Belongs to [0,1]
        merge_thresh    : used in non maxima suppression to caracterise overlap allowed.
        verbose         : if you want an estimated time when doing without video processing. Overhead is minimal according to tqdm
        output_path     : the path where we output the video if not None
        identifier      : nothing for the moment. Coming Soon.
        interactive     : if true it shows in "realtime" the result of the processing. Make it slower, but can be use to QC
        fps             : maximal speed of the interactive. Often the maximal speed comes from the processing...

        """
        if scenes is not None:
            self.scenes = scenes
        if self.scenes is None:
            print("Compute scenes.")
            self.find_scenes()
        # we compute the frame where we will have to do detection.
        if verbose:
            print("Compute frame number to be analysed by detector.")
        frames_where_detect = _get_detection_frames(rate, self.scenes, self.max_frame, verbose)
        

        #MIN_CONF = threshold_detect
        #NMS_THRESH = merge_thresh
        #object_counter = 0
    
        def inner_func(img, frame_nb, *ctx):

            # context loading
            if ctx != ():
                boxes=ctx[0]
                trackables=ctx[1]
                object_counter = ctx[2]
            else:
                # initialise context if it does not exist
                trackables=dict()
                object_counter = 0

            # detect, identify, track 
            if frame_nb in frames_where_detect:
                boxes = _detect(img, detector, threshold_detect, merge_thresh)
                trackables = _identify(img, boxes, trackables, object_counter, frame_nb, tracker_type)
                object_counter+=len(boxes)
            else:
                _track(img, trackables)

            # draw the boxes
            for key, item in trackables.items():
                img = item.draw_rectangle(img)
        
            return img, (ctx,trackables,object_counter)
        if not interactive:
            fps = -1 #go as fast as possible
            window_name = None
        else:
            window_name = "Detect Human"

        self.custom_process(func=inner_func, reset=True, fps=fps, release_end=False, window_name=window_name, output_path=output_path, max_frame=max_frame)

    def cut_scenes(self, release_end=False, reset=True):
        if reset:
            self.reset()
        frame_counter = 0
        pbar = tqdm.tqdm(total=self.max_frame, unit=" frames ", smoothing=0.1)
        if self.by_scene != []:
            return
        # scene are between scene[k][0] and scene[k][1]-1 => nb_frame = 
        self.by_scene = [np.zeros((self.scenes[k][1].get_frames()-self.scenes[k][0].get_frames(),self.h,self.w,3), dtype=np.uint8) for k in range(len(self.scenes))]
        for sc_counter,scene in enumerate(self.scenes):
            counter = 0
            while frame_counter < scene[1].get_frames():
                ok, frame = self.get_next_frame()
                if not ok:
                    # well should be because we finish the video?
                    break
                self.by_scene[sc_counter][counter,:,:,:]=frame
                frame_counter += 1
                counter+=1
                pbar.update(1)
        if release_end:
            self.release()
        pbar.close()

    def write(self, video_path, stats=None, look_result=False, max_frame=-1, release_end=False):
        window_name = None
        if look_result:
            window_name = "LOOOK !"
        func=None
        if stats is not None:
            def func(img, frame_nb, *ctx):
                if ctx !=():
                    scene_counter = ctx[0]
                else:
                    scene_counter = 0
                
                trackables = stats[scene_counter][frame_nb-self.scenes[scene_counter][0].get_frames()]["trackables"]
                for key, item in trackables.items():
                    box = item["box"]
                    p1 = (int(box[0]), int(box[1]))
                    p2 = (int(box[2]+box[0]), int(box[3]+box[1]))
                    cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
                    # img = item.draw_rectangle(img)
                if self.scenes[scene_counter][-1].get_frames()-1 == frame_nb:
                    scene_counter+=1
                return img, (scene_counter,)
        self.custom_process(output_path=video_path, verbose=True, func=func,reset=True, release_end=release_end, window_name=window_name,
                        ctrl_func=None, dyn_ctrl_fps=True,fps=-1, max_frame=max_frame)

    # DOES NOT WORK AT ALL IN MULTIPROCESSING! THIS APPROACH IS TOO SIMPLE AND THE MIX WITH TORCH/OPENCV AND EVERYTHING CREATE STRANGE THING.
    def compute_stat(self, detector, rate, scenes=None, tracker_type="csrt", verbose=True, output_path=None, identifier=None, max_frame=-1,
    threshold_detect=0.8, merge_thresh=0.3, cpu_count=None, video_path=None, look_result=False, release_end=True, cache=True):
        """
        Use the tracker and detector given as parameter to create a new video.

        Args
        ----
            detector        : an object Detector. Will detect the element according to the setting of the object.
            rate            : if no scenes are found, do detection every rate frames.
            scenes          : the scenes , overwrite self.scenes if not NONE, if scene and self.scene are none we compute the scene
            tracker_type    : the different opencv implemented tracker, value in ["csrt","kcf","boosting","mil","tld","medianflow","mosse"]
            verbose         : if you want an estimated time when doing without video processing. Overhead is minimal according to tqdm
            output_path     : the path where we output the video if not None
            identifier      : nothing for the moment. Coming Soon.

        """
        # TODO : Implement the cache...
        if self.by_scene==[]:
            self.cut_scenes()
        if max_frame<0:
            max_frame = self.max_frame
        if cpu_count is None:
            cpu_count = mp.cpu_count()*2
        scene_stop = len(self.scenes)
        frames_where_detect_by_scene = _get_detection_frames(rate, self.scenes, self.max_frame, verbose, cut_scene=True)
        if verbose:
            new_max_frame=0
            scene_stop=0
            for scene_counter in range(len(self.scenes)):
                new_max_frame+=(self.scenes[scene_counter][1].get_frames()-self.scenes[scene_counter][0].get_frames())
                if new_max_frame>max_frame:
                    scene_stop = scene_counter
                    max_frame=new_max_frame
                    break
            if scene_stop==0:
                scene_stop=len(self.scenes)
            else:
                scene_stop+=1

            pbar = tqdm.tqdm(total=max_frame, unit=" frames ", smoothing=0.1)
        # we define the func to be called in map
        def inner_func(info):
            scene_img, frames_where_detect = info

            stats = dict()
            # we want to keep for each object we detect the trajectory of boxs, conf, labels
            object_counter = 0
            trackables=dict()
            for frame_nb, img in enumerate(scene_img):
                if verbose:
                    # does not work in multiprocessing
                    pbar.update(1)
                stats[frame_nb] = {}
                # detect, identify, track 
                if frame_nb in frames_where_detect:
                    all_boxes = []
                    all_conf = []
                    to_del = []
                    for tracked in trackables:
                        # if we didnt lose the track we just update it via un detector
                        
                        trackables[tracked].detect(img, detector, threshold_detect)
                        if trackables[tracked].box is None:
                            to_del+=[tracked]
                        else:
                            all_boxes += [list(trackables[tracked].dnn_box())]
                            all_conf+=[1]
                    for todel in to_del:
                        del trackables[todel]
                    boxes, conf, labels, true_boxes = _detect(img, detector, threshold_detect, merge_thresh, ret_all=True)
                    # we add all the new boxes we found after non maxima suppression
                    all_boxes += [[box[0],box[1],box[2],box[3]] for box in boxes]
                    
                    all_conf+=[cf for cf in conf]
                    all_boxes = np.array(all_boxes, dtype=np.int32)
                    all_conf = np.array(all_conf, dtype=np.int32)
                    idxs = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_conf.tolist(), threshold_detect, merge_thresh)
                    if len(idxs)>0:
                        boxes = all_boxes[idxs.flatten()]
                        conf = all_conf[idxs.flatten()]
                    if len(boxes)>len(trackables):
                        z_fix = len(trackables)
                        for k in range(len(boxes)-z_fix):
                            trackables[object_counter+k] = TrackableObject(object_counter+k,img=img, box=boxes[k+z_fix], frame_nb=frame_nb, tracker_type=tracker_type)
                        object_counter+=(len(boxes)-z_fix)
                    # trackables = _identify(img, boxes, trackables, object_counter, frame_nb, tracker_type)
                    #object_counter+=len(boxes)
                    stats[frame_nb]["conf"]=conf
                    stats[frame_nb]["trackables"]=copy.deepcopy({key:item.info() for key,item in trackables.items()})
                    stats[frame_nb]["boxes"]=true_boxes
                    stats[frame_nb]["labels"]=labels
                else:
                    _track(img, trackables)
                    stats[frame_nb]["trackables"]=copy.deepcopy({key:item.info() for key,item in trackables.items()})
            
            return stats
        to_inject = [(scene_img, frames_where_detect) for scene_img, frames_where_detect in zip(self.by_scene[:scene_stop],frames_where_detect_by_scene[:scene_stop])]
        
            
        
        #ret = dill_map(cpu_count, inner_func, to_inject) -> :(
        ret = [inner_func(info) for info in to_inject]
        if verbose:
            pbar.close()
        
        if not(video_path is None and not(look_result)):
            self.write(video_path, stats=ret, look_result=look_result, max_frame=max_frame, release_end=True)
        if output_path is not None or cache:
            if output_path is None:
                output_path = os.path.join(os.path.dirname(video_path),os.path.basename(video_path)+".evia")
            with open(output_path,'wb') as f:
                cp.dump(ret, f)
        return ret


        
