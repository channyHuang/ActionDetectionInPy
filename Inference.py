import cv2
import numpy as np
import threading
import torch

from models.experimental import attempt_load
from bytetrack_utils.byte_tracker_new import BYTETracker
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
# STGCN
from stgcn.ActionsEstLoader import TSSTG

from config import *

# kpts = del_tensor_ele(kpts, 3, 15)
def del_tensor_ele(arr, index_a, index_b):
    arr1 = arr[0:index_a]
    arr2 = arr[index_b:]
    return torch.cat((arr1, arr2), dim=0)

def tlwh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y

# ct+++   action_model, frame
def track_main(tracker, detection_results, Result_kpts, frame_id, image_height, image_width, test_size,
               action_model, frame):
    '''
    main function for tracking
    :param args: the input arguments, mainly about track_thresh, track_buffer, match_thresh
    :param detection_results: the detection bounds results, a list of [x1, y1, x2, y2, score]
    :param frame_id: the current frame id
    :param image_height: the height of the image
    :param image_width: the width of the image
    :param test_size: the size of the inference model
    '''
    online_targets = tracker.update(detection_results, Result_kpts, [image_height, image_width], test_size)
    online_tlwhs = []
    online_ids = []
    online_scores = []
    results = []
    aspect_ratio_thresh = 1.6  # +++++
    min_box_area = 10  # ++++
    action_results = []

    for target in online_targets:
        tlwh = target.tlwh
        tid = target.track_id
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area or vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(target.score)
            # save results
            results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{target.score:.2f},-1,-1,-1\n"
                    )

        action = 'pending..'
        clr = (0, 255, 0)
        # Use 30 frames time-steps to prediction.
        if len(target.keypoints_list) == 30:
            pts = np.array(target.keypoints_list, dtype=np.float32)
            out = action_model.predict(pts, frame.shape[:2])
            action_name = action_model.class_names[out[0].argmax()]
            action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
            if action_name == 'Fall Down':
                clr = (255, 0, 0)
            elif action_name == 'Lying Down':
                clr = (255, 200, 0)
            # print(action)
            action = action
        action_results.append(action)

    return online_tlwhs, online_ids, action_results  # ct+++action_results

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class Inference():
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Inference, "_instance"):
            with Inference._instance_lock:
                if not hasattr(Inference, "_instance"):
                    Inference._instance = object.__new__(cls)
        return Inference._instance
    
    def init(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != 'cpu'
        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        # Tracker
        self.tracker = BYTETracker(track_thresh, track_buffer, match_thresh)  # ct+++
        # Actions Estimate.
        self.action_model = TSSTG()

        self.model.half()  # to FP16
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        self.frame_id = 0

    def detect(self, frame):
        # Padded resize

        img = letterbox(frame, (imgsz, imgsz), stride=32, auto=False)[0]
        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres,  iou_thres, classes=filter_classes, agnostic=agnostic_nms, kpt_label=kpt_label)

        # Process detections
        for i, det in enumerate(pred): 
            # det 包括51个关键点结果(x,y,cof)，6个box结果(xyxy,cof,cla)
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], frame.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], frame.shape, kpt_label=kpt_label, step=3)

                Results = []  # ct+++
                Result_kpts = []

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    Results.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf)])  # ct+++

                    # [[x,y,c],[x,y,c]...]
                    Result_kpts.append(del_tensor_ele(det[det_index, 6:], 3, 15).reshape(-1, 3))  # 存储13个kpt
                # print(Result_kpts)

                online_tlwhs, online_ids, action_results = track_main(self.tracker, np.array(Results), Result_kpts, self.frame_id,
                                                                      1080, 1920,
                                                                        (1080, 1920), self.action_model, frame)  # ct+++
                online_tlwhs = np.array(online_tlwhs).reshape(-1, 4)
                online_xyxys = tlwh2xyxy(online_tlwhs)
                outputs = np.concatenate((online_xyxys, np.array(online_ids).reshape(-1, 1)), axis=1)

                # Write results
                for det_index, (output, conf) in enumerate(zip(outputs, det[:, 4])):

                    xyxy = output[0:4]
                    id = output[4]

                    # Add bbox to image
                    label = f'{int(id)} {conf:.2f} {str(action_results[det_index])}'
                    kpts = det[det_index, 6:]

                    plot_one_box(xyxy, frame, label=label, color=None,
                                    line_thickness=3, kpt_label=kpt_label, kpts=kpts, steps=3,
                                    orig_shape=frame.shape[:2])
        self.frame_id += 1
        return frame


inference = Inference()

if __name__ == '__main__':
    inference.init()
    cap = cv2.VideoCapture('video.avi')
    while cap.isOpened():
        res, frame = cap.read()
        if res == False:
            break
        frame = inference.detect(frame)
        cv2.imshow('res', frame)
        cv2.waitKey(1)  # 1 millisecond
    cv2.destroyAllWindows()