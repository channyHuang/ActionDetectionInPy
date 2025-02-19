from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

class ThreadPool():
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(ThreadPool, "_instance"):
            with ThreadPool._instance_lock:
                if not hasattr(ThreadPool, "_instance"):
                    ThreadPool._instance = object.__new__(cls)
        return ThreadPool._instance
    
    def __init__(self):
        self.num_thread = 3
        self.alive = False
        self.queue = Queue()

    def startThreads(self, num_thread, func):
        if self.alive == True:
            return
        self.alive = True
        self.num_thread = num_thread
        self.pool = ThreadPoolExecutor(max_workers=num_thread)
        self.func = func

    def stopThreads(self):
        if self.alive == False:
            return
        self.alive = False
        self.pool.shutdown()

    def initModel(self, model_path):
        self.model = None

    def put(self, stSyncData):
        self.queue.put(self.pool.submit(self.func, stSyncData))

    def get(self):
        if self.queue.empty():
            return None, False
        temp = []
        temp.append(self.queue.get())
        for data in as_completed(temp):
            return data.result(), True
        
threadPool = ThreadPool()

if __name__ == '__main__':
    from Inference import *

    def func(stSyncData):
        stSyncData.res = inference.detect(stSyncData.sub_frame)
        return stSyncData
    
    inference.init()
    threadPool.startThreads(3, func)
    cap = cv2.VideoCapture('video.avi')
    while cap.isOpened():
        res, frame = cap.read()
        if res == False:
            break
        # stSyncData = StSyncData(sub_frame, frame_id, i)
        # threadPool.put(stSyncData)
    
    threadPool.stopThreads()