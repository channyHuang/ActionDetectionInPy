import cv2
import time
import math
import multiprocessing

from StreamInC import streamInC
from ThreadPool import threadPool

from config import *
from Inference import *

def detect(video = 'video.avi'):
    inference.init()
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        # frame = streamInC.getFrame()
        res, frame = cap.read()
        if res == False:
            break
        frame = cv2.resize(frame, (1920, 1080))
        frame = inference.detect(frame)
        cv2.imshow('res', frame)
        cv2.waitKey(1)  # 1 millisecond

def detect_stream(url = 'rtsp://admin:@192.168.1.155'):
    streamInC.start(url)
    time.sleep(10)
    while True:
        picture = streamInC.getFrame()
        if picture is None:
            time.sleep(0.1)
            continue

        frame = inference.detect(picture)

        cv2.imshow('res', frame)
        cv2.waitKey(1)

# 分片计算子图像offset, crop_size = [height, width]
def cal_offset(shape=[1080, 1920], crop_size=[1080, 1920]):
    height_num = math.ceil(shape[0] / crop_size[0])
    width_num = math.ceil(shape[1] / crop_size[1])
    height_pad, width_pad = 0, 0
    if height_num > 1:
        height_pad = (height_num * crop_size[0] - shape[0]) // (height_num - 1)
    if width_num > 1:
        width_pad = (width_num * crop_size[1] - shape[1]) // (width_num - 1)
    height_offset = 0
    width_offset = 0
    offset = []
    edge_flag = [False, False]
    while height_offset < shape[0]:
        edge_flag[1] = False
        if height_offset + crop_size[0] >= shape[0]:
            height_offset = shape[0] - crop_size[0]
            edge_flag[0] = True
        while width_offset < shape[1]:
            if width_offset + crop_size[1] > shape[1]:
                width_offset = shape[1] - crop_size[1]
                edge_flag[1] = True
            offset.append([height_offset, width_offset])
            if edge_flag[1]:
                break
            width_offset += crop_size[1] - width_pad
        if edge_flag[0]:
            break
        height_offset += crop_size[0] - height_pad
        width_offset = 0
    return offset

class StSyncData:
    def __init__(self, sub_frame, frame_id, id):
        self.sub_frame = sub_frame
        self.frame_id = frame_id
        self.id = id
        self.res = None

def thread_func(stSyncData):
    stSyncData.res = inference.detect(stSyncData.sub_frame)
    return stSyncData

def detect_crop(url = 'video.avi'):
    inference.init()
    threadPool.startThreads(3, thread_func)

    send_queue = multiprocessing.Queue(maxsize=100)

    spend_time = "30 frame average fps:"
    loopTime = time.time()
    
    cap = cv2.VideoCapture(url)
    frame_id = 0
    count = 0
    offset = None
    offset_end = []
    while cap.isOpened():
        res, frame = cap.read()
        if res == False:
            break
        
        # 计算切片子图像相对于整张图像的偏移
        if offset is None:
            shape = frame.shape
            offset = cal_offset(shape, [crop_size[0], crop_size[1]])
            block = len(offset)
            offset_end = [[offset[i][0] + crop_size[0], offset[i][1] + crop_size[1]] for i in
                            range(block)]
            print('clip info:', shape, block, offset, offset_end)

        # 把每张子图像放入线程池
        for i in range(block):
            sub_frame = frame[offset[i][0]:offset_end[i][0], offset[i][1]:offset_end[i][1]]
            stSyncData = StSyncData(sub_frame, frame_id, i)
            threadPool.put(stSyncData)

        count += block
        frame_id += 1

        # 获取推理结果，拼接子图像恢复到原始图像大小
        final_img = frame
        for _ in range(block):
            stSyncData_get, flag_pool_get = threadPool.get()
            if flag_pool_get == False:
                time.sleep(0.002)
                continue
            i = stSyncData_get.id
            final_img[offset[i][0]:offset_end[i][0], offset[i][1]:offset_end[i][1], :] = stSyncData_get.res[:,:,:]
        
        if frame_id >= 30:
            spend_time = "30 frame average fps: {:.2f}".format(round(frame_id / (time.time() - loopTime), 2))
            frame_id = 0
        cv2.putText(final_img, spend_time, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('res', final_img)
        cv2.waitKey(1)
    threadPool.stopThreads()

if __name__ == '__main__':
    # detect('rtsp://admin:@192.168.1.155')
    # detect_stream()
    detect_crop('rtsp://admin:@192.168.1.155')