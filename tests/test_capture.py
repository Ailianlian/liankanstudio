from liankanstudio.capture import _get_detection_frames

class mockscene(object):
    def __init__(self, frame):
        self.frame = frame
    
    def get_frames(self):
        return self.frame
#_get_detection_frames(rate, scenes, max_frame, verbose, cut_scene=False)

def test_get_detection_frames():
    rate = 5
    scenes = [(mockscene(0),mockscene(4)),
                (mockscene(4),mockscene(11)),
                (mockscene(11),mockscene(15))]
    max_frame = 15
    verbose = False
    # if cut_scene is false we should return [0,5,10,12]
    assert _get_detection_frames(rate, scenes, max_frame, verbose, cut_scene=False) == [0,4,9,11]

def test_get_detection_frames_cut_scene():
    rate = 5
    scenes = [(mockscene(0),mockscene(4)),
                (mockscene(4),mockscene(11)),
                (mockscene(11),mockscene(15))]
    max_frame = 15
    verbose = False
    # if cut_scene is true we should return [[0],[0,5],[0]]
    assert _get_detection_frames(rate, scenes, max_frame, verbose, cut_scene=True) == [[0],[0,5],[0]]