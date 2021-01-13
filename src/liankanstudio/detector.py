"""
File containing the detector, for one image
"""
import cv2
import os
import numpy as np
root = os.path.dirname(os.path.abspath(__file__))
def get_data(rel_path):
    return os.path.join(root,rel_path)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
CV2_MAJOR_VER = int(major_ver)
CV2_MINOR_VER = int(minor_ver)
try:
    import torchvision # to use pretrained model
    has_torchvision=True
    import torchvision.transforms as T
    import torch
except:
    print("Torchvision model not supported.")
    has_torchvision=False

SSD_CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def _dnn_preprocess(src, size=300, scale=1/127.5, mean=127.5):
    #img = cv2.resize(src, (300,300))
    #img = img - 127.5
    #img = img * 0.007843
    #img = img.astype(np.float32)
    #img = img.transpose((2, 0, 1))
    img = cv2.dnn.blobFromImage(src,scale,(size,size),mean)
    return img


def _yolo_postprocess(img, out):
    outputs = np.vstack(out)
    h = img.shape[0]
    w = img.shape[1]

    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        x, y, u, v = output[:4] * np.array([w, h, w, h])
        boxes.append([int(x),int(y),int(x+u), int(y+v)])
        confidences.append(float(confidence))
        classIDs.append(classID)
        # cv.rectangle(img, p0, p1, WHITE, 1)
    return (boxes, confidences,np.array([COCO_INSTANCE_CATEGORY_NAMES[i] for i in classIDs if COCO_INSTANCE_CATEGORY_NAMES[i]] ))

def _ssd_postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    
    labels = np.array([SSD_CLASSES[int(i)] for  i in out[0,0,:,1]])
    box = out[0,0,:,3:7] * np.array([w, h, w, h])
    conf = out[0,0,:,2]
    return (box.astype(np.int32), conf, labels)

class Detector(object):

    def _define_torch_model(self,torchname, **kwargs):
        if has_torchvision:
            if torchname=="torchfrcnn":
                torchfrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            elif torchname=="torchmrcnn":
                torchfrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            else:
                torchfrcnn_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            torchfrcnn_model.to(device)
            torchfrcnn_model.eval()
            self.device = device
            self.model = torchfrcnn_model
            self.classes = COCO_INSTANCE_CATEGORY_NAMES
            self.downfact = kwargs.get("downfact",4)
        else:
            print("Torchvision model not supported.")

    def _post_process_torch(self,img, torchname):
        h,w = img.shape[0:2]
        transform = T.Compose([T.ToTensor(),T.Resize(size=(int(h/self.downfact),int(w/self.downfact)))])
        img_process = transform(img)
        img_process = img_process.to(device=self.device)
        out = self.model([img_process])[0]
        if torch.cuda.is_available():
            return self._dnn_target(out["boxes"].detach().cpu().numpy()*self.downfact, out['scores'].detach().cpu().numpy(), 
            np.array([COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(out['labels'].cpu().numpy()) if COCO_INSTANCE_CATEGORY_NAMES[i]]))
        else:
            return self._dnn_target(out["boxes"].detach().numpy()*self.downfact, out['scores'].detach().numpy(), 
            np.array([COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(out['labels'].numpy()) if COCO_INSTANCE_CATEGORY_NAMES[i]]))


    def __init__(self, method, target="person",*args, **kwargs):
        self.target = target
        self.method = method
        self.param = args
        if method=="yolo":
            model_proto_path = kwargs.get("proto_path",get_data(os.path.join("data","yolo","yolo.cfg")))
            model_weights = kwargs.get("weights",get_data(os.path.join("data","yolo","yolo.weights")))
            self.classes = kwargs.get("classes",SSD_CLASSES)
            self.model = cv2.dnn.readNetFromDarknet(model_proto_path, model_weights)
            self.size = kwargs.get("size",300)
            self.postprocess = kwargs.get("postprocess", lambda x:x)
        elif "torch"in method:
            self._define_torch_model(method, **kwargs)
        elif method=="ssd":
            # weights from here https://github.com/chuanqi305/MobileNet-SSD
            # mobilenet backbone so this is quite fast but not as accurate as other model.
            model_dir = kwargs.get("model_dir","./")
            print(get_data(os.path.join("data","ssd","ssd_deploy.prototxt")))
            model_proto_path = kwargs.get("proto_path",get_data(os.path.join("data","ssd","ssd_deploy.prototxt")))
            model_weights = kwargs.get("weights",get_data(os.path.join("data","ssd","ssd_deploy.caffemodel")))
            self.classes = kwargs.get("classes",SSD_CLASSES)
            self.model = cv2.dnn.readNetFromCaffe(model_proto_path, model_weights)
        elif method=="haar":
            if self.param == []:
                self.param = (1.1, 3)
            # we simply load the cascade classifier in opencv-python system
            # for this method we allow target in [eyes, smile, eye_glasses, catface, face, fullbody, upperbody, lowerbody, licence_plate]
            # this method is really bad if we don't help it.
            if target=="eye":
                self.detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_eye.xml"))
            elif target=="smile":
                self.detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_smile.xml"))
            elif target=="eye_glasses":
                self.detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_eye_tree_eyeglasses.xml"))
            elif target=="catface":
                self.detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_frontalcatface.xml"))
            elif target=="face":
                self.detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_frontalface_alt.xml"))
            elif target=="fullbody":
                self.detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_fullbody.xml"))
            elif target=="upperbody":
                self.detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_upperbody.xml"))
            elif target=="lowerbody":
                self.detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_lowerbody.xml"))
            elif target=="licence_plate":
                self.detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_licence_plate_rus_16stages.xml"))
            elif target=="person":
                self.detector = [cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_eye.xml")),
                   cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_fullbody.xml")),
                   cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_frontalface_alt.xml")),
                   cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_upperbody.xml")),
                   cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_lowerbody.xml"))]
            else:
                print("This method does not work with this target")
        elif method=="human":
            window_name = kwargs.get("window_name","Select ROIs")
            self.detector = lambda img : cv2.selectROIs(window_name,img)
        else:
            raise AttributeError("No method have been selected")

    def _dnn_target(self, boxs, conf, labels):
        mask = labels==self.target
        if isinstance(mask, bool):
            if mask:
                return boxs, conf, labels
            else:
                return [], [], []
        if sum(mask)==0:
            return [], [], []
        else:
            return boxs[mask], conf[mask], labels[mask]

    def detect(self, img):
        if self.method=="yolo":
            img_process = _dnn_preprocess(img, scale=1/255, mean=0,size=412)
            self.model.setInput(img_process)
            boxs, conf, labels = _yolo_postprocess(img, self.model.forward()) 
            return self._dnn_target(boxs, conf, labels)
        elif "torch" in self.method:
            return self._post_process_torch(img, self.method)
        elif self.method=="ssd":
            img_process = _dnn_preprocess(img)
            self.model.setInput(img_process)
            boxs, conf, labels = _ssd_postprocess(img, self.model.forward()) 
            return self._dnn_target(boxs, conf, labels)
        elif self.method=="haar":
            # we simply load the cascade classifier in opencv-python system
            # for this method we allow target in [eyes, smile, eye_glasses, catface, face, fullbody, upperbody, lowerbody, licence_plate]
            
            if self.target=="person":
                image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                 # detect "all" the targets in the image
                boxs=[]
                for detect in self.detector:
                    boxs+=[*(detect.detectMultiScale(image_gray, *(self.param)))]
            else:
                boxs = self.detector.detectMultiScale(image_gray, *(self.param))
            for k in range(len(boxs)):
                boxs[k]=[boxs[k][0],boxs[k][1],boxs[k][0]+boxs[k][2],boxs[k][1]+boxs[k][3]]
            boxs=np.array(boxs)
            conf = np.ones(len(boxs))
            return boxs,conf,None
        elif self.method=="human":
            boxes = self.detector(img)
            conf = np.ones(len(boxes))
            # the boxes here are x,y,h,w but in dnn this is x1,y1,x2,y2
            for k in range(len(boxes)):
                boxes[k]=[boxes[k][0],boxes[k][1],boxes[k][0]+boxes[k][2],boxes[k][1]+boxes[k][3]]
            boxes=np.array(boxes)
            return boxes,conf,None
        else:
            # should not happen
            raise AttributeError("No method have been selected")