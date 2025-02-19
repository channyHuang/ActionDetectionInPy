# model
weights = 'Yolov5s6_pose_1280_ti_lite.pt'

# detect
conf_thres = 0.1
iou_thres = 0.45
filter_classes = None
agnostic_nms = False
kpt_label = True

# track
track_thresh = 0.5
track_buffer = 30
match_thresh = 0.8
imgsz = 1280 # //32

# crop
# [height, width]
crop_size = [1280, 1280]
block = 1