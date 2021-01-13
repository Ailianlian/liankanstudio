import numpy as np
import cv2
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
CV2_MAJOR_VER = int(major_ver)
CV2_MINOR_VER = int(minor_ver)
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}


class TrackerSet(object):
    """
    It tracks in a image but with multiple tracker for one object and mixing method when more than one tracker used.
    
    Should be used with multiprocessing to make it more efficient.

    cv2.MultiTracker_create() exists, maybe better. This class is mainly for learning purpose for the moment.
    """
    
    def _add(self, tracker_type):
        # tracker_types = {'BOOSTING':0, 'MIL':1,'KCF':2, 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        # BOOSTING tracker is very soso, slow and bad since try to always find something need scene_cut at minimum
        # MIL tracker will alaways try to find something and is quite slow. Very good with occlusion if we are sure the thing stay in image...
        # KCF very fast but needs more detection rate and does not recover from occlusion
        # TLD tracker is very soso, i don't know the stat but never use it 
        # MEDIANFLOW is very fast but very bad at tracking failure and at identification of common thing
        # GOTURN fail, needs to fix some issue with it... so we won't use it
        # MOSSE is very fast, seems quite ok with occlusion? at least this is very fast can be good for fast tracking
        # CSRT is soso too, middle fast.
        if tracker_type not in OPENCV_OBJECT_TRACKERS:
            raise AttributeError("tracker_type does not exist")
        if int(self.major_ver) < 4 and int(self.minor_ver) < 3:
            tracker = cv2.cv2.Tracker_create(tracker_type)
        else:
            tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()
        return tracker
                
    def create_tracker(self, tracker_types):
        """
        tracker_types can be a list of tracker type.
    
        We will use the list as a tracker.
        """
        # TODO : add the ability to create your own tracker based on your own implementation? Should be possible in opencv but how?
        self.trackers =  [self._add(tracker_type) for tracker_type in self.tracker_types]
    
    def __init__(self, tracker_types):
        self.major_ver = CV2_MAJOR_VER
        self.minor_ver = CV2_MINOR_VER
        self.tracker_types = tracker_types
        self.create_tracker()

    def init(self, img, bbox):
        for tracker in self.trackers:
            ok = tracker.init(img, bbox)
            if not ok:
                return False
        return True
    
    def update(self, img, method="average"):
        # TODO : add a choosen method to take choose one box in the end.
        bboxs = []
        box_counter = 0
        ok = True
        for tracker in self.trackers:
            temp_ok, bbox = tracker.update(img)
            if len(bbox)>0:
                bboxs+=[bbox]
            box_counter+=1
            ok = temp_ok and ok
        if method=="average":
            bbox = np.array(bboxs)/box_counter
            ok = box_counter>(len(self.trackers)/2) and ok
        return ok, bbox

    def set_types(self, new_tracker_types):
        self.tracker_types = new_tracker_types

    def reset(self):
        # Not sure this is useful
        self.trackers = []
        self.create_tracker()


class TrackableObject(object):
    """
    A trackable object instance, keep when the object was detected, the last box about it, etc.
    """
    def __init__(self, obj_id, img=None, frame_nb=None, box=None, tracker_type=None, tracker_types=None):
        self.last_detect = None
        self.obj_id = obj_id
        self.major_ver = CV2_MAJOR_VER
        self.minor_ver = CV2_MINOR_VER
        self.is_updated = False
        self.tracker=None
        self._is_tracked = False
        self.tracker_type = tracker_type
        self.box=None
        if img is not None and frame_nb is not None and box is not None:
            if tracker_type is not None:
                self.init_fast_tracker(frame_nb, box, img, tracker_type=tracker_type)
            elif tracker_types is not None:
                self.init_fast_tracker(frame_nb, box, img, tracker_types=tracker_types)
            else:
                self.init_fast_tracker(frame_nb, box, img)
            
    def info(self):
        return {"tracker_type":self.tracker_type,"obj_id":self.obj_id,"box":self.box }

        
    def init_tracker(self, frame_nb, box, frame, tracker_types=["KCF"]):
        self.last_detect = frame_nb
        self.tracker = TrackerSet(tracker_types, self.major_ver, self.minor_ver)
        ok = self.tracker.init(frame, tuple((box[0],box[1],box[2]-box[0],box[3]-box[1])))
        self._is_tracked = ok
        self.box = tuple((box[0],box[1],box[2]-box[0],box[3]-box[1]))
        return ok
    
    def init_fast_tracker(self, frame_nb, box, frame, tracker_type="mosse"):
        self.last_detect = frame_nb
        self.tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()
        ok = self.tracker.init(frame, tuple((box[0],box[1],box[2]-box[0],box[3]-box[1])))
        self._is_tracked = ok
        self.box = tuple((box[0],box[1],box[2]-box[0],box[3]-box[1]))
        return ok
    
    def update_box(self, frame):
        ok, box = self.tracker.update(frame)
        self.is_updated = ok
        self._is_tracked = ok
        self.box = box
        return box

    def is_tracked():
        return self._is_tracked

    def draw_rectangle(self, img):
        p1 = (int(self.box[0]), int(self.box[1]))
        p2 = (int(self.box[2]+self.box[0]), int(self.box[3]+self.box[1]))
        cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
        cv2.putText(img, f"Object {self.obj_id}.", (p1[0],p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)
        return img

    def detect(self, img, detector, threshold_detect):
        h,w = img.shape[0:2]
        crop = img[int(max(self.box[0]-50,0)):int(min(self.box[2]+self.box[0]+50,h)),int(max(self.box[1]-50,0)):int(min(self.box[3]+self.box[1]+50,w))]
        boxes, conf, labels = detector.detect(img)
        if len(boxes)>0:
            #higher confidence one normally?
            box=boxes[0]
            self.box = tuple((int(max(self.box[0]-50,0))+box[0],int(max(self.box[1]-50,0))+box[1],box[2]-box[0],box[3]-box[1]))
        else:
            self.box = None
    
    def dnn_box(self):
        return [int(self.box[0]),int(self.box[1]),int(self.box[2]+self.box[0]),int(self.box[3]+self.box[1])]