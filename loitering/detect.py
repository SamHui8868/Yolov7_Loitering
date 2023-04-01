
import sys
sys.path.append("./loitering/yolov7_quick_start/")
import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression,check_img_size

class Yolov7_Detector():
    def __init__(self,weight_pth="./loitering/yolov7_quick_start/weights/yolov7.pt",inference_device="cpu",img_size=640,conf_thresh=0.25, iou_thresh=0.45):
        self.weights=weight_pth
        self.device=inference_device
        self.img_size=img_size
        self.conf_thresh=conf_thresh
        self.iou_thresh=iou_thresh
    # Load Yolov7 model using attempt_load() in Yolov7 src code
    def load_model(self):
        self.model = attempt_load(self.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size


    def detect(self,src_image):
        conf_thresh=self.conf_thresh
        iou_thresh=self.iou_thresh
        image = np.asarray(src_image).copy()
        # Resize the image to match the inference size
        h_org,w_org = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Transform image from numpy to torch format
        image_pt = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        # Normalize to 0-1
        image_pt = image_pt.float() / 255.0
        
        # Inference
        with torch.no_grad():# Computing Gradients will cause memory leak
            pred = self.model(image_pt[None], augment=False)[0]
            
        # Non-Maximum Suppression
        res= non_max_suppression(pred,conf_thres=conf_thresh, iou_thres=iou_thresh, classes=0, agnostic=False)[0].cpu().numpy()
        
        # Resize boxes to the original resolution
        res[:, [0, 2]] *= w_org / self.img_size
        res[:, [1, 3]] *= h_org / self.img_size
        
        return res

#Test the detector with an image
if __name__ == '__main__':
    class_names=np.load("./loitering/yolov7_quick_start/coco_80class.npy")
    # Function to draw bounding box on the detected object with class name
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

        label = str(class_names[int(class_id)])

        color = COLORS[int(class_id)]

        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

        cv2.putText(img, label+' '+str(int(confidence)), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    
    # Load data
    img_pth = "./loitering/yolov7_quick_start/test_imgs/bus.jpg"
    image = cv2.imread(img_pth)
    
    # Generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    
    # Construct Detector Class   
    # inference device: for cuda device specify the device id 'cuda:0'  for CPU 'cpu'
    yolov7=Yolov7_Detector(inference_device='cpu')
    yolov7.load_model()
    # Predict
    pred = yolov7.detect(image)
    
    # Visulaize Result
    for x1, y1, x2, y2, conf, class_id in pred:
        draw_bounding_box(image,int(class_id),conf,int(x1),int(y1),int(x2),int(y2))
    cv2.imshow("Result",image)
    cv2.waitKey(0)




