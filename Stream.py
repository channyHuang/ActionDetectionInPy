
import cv2
import multiprocessing
import subprocess
import time
import threading

def rtsp_stream_decode(uri, width, height, latency, Decoder='H264'):
    """Open an RTSP URI (IP CAM)."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'omxh264dec' in gst_elements:
        # Use hardware H.264 decoder on Jetson platforms
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! videoconvert ! '
                   'appsink').format(uri, latency, width, height)
    elif 'avdec_h264' in gst_elements:
        # Otherwise try to use the software decoder 'avdec_h264'
        # NOTE: in case resizing images is necessary, try adding
        #       a 'videoscale' into the pipeline
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! avdec_h264 ! '
                   'videoconvert ! appsink').format(uri, latency)
    elif 'omxh265dec' in gst_elements:
        # Use hardware H.264 decoder on Jetson platforms
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph265depay ! h265parse ! omxh265dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! videoconvert ! '
                   'appsink').format(uri, latency, width, height)
    elif 'avdec_h265' in gst_elements:
        # Otherwise try to use the software decoder 'avdec_h264'
        # NOTE: in case resizing images is necessary, try adding
        #       a 'videoscale' into the pipeline
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph265depay ! h265parse ! avdec_h265 ! '
                   'videoconvert ! appsink').format(uri, latency)
    elif 'nvv4l2decoder' in gst_elements:
        # Otherwise try to use the software decoder 'avdec_h264'
        # NOTE: in case resizing images is necessary, try adding
        #       a 'videoscale' into the pipeline
        if Decoder == 'H264':
            gst_str = (
                'rtspsrc location={} latency={} ! '
                'rtph264depay ! h264parse ! nvv4l2decoder ! '
                'nvvidconv ! '
                'video/x-raw, width={}, height={},format=BGRx ! '
                'videoconvert !'
                'video/x-raw, format=BGR ! '
                'appsink').format(uri, latency, width, height)
        elif Decoder == 'H265':
            gst_str = (
                'rtspsrc location={} latency={} ! '
                'rtph265depay ! h265parse ! nvv4l2decoder ! '
                'nvvidconv ! '
                'video/x-raw, width={}, height={},format=BGRx ! '
                'videoconvert !'
                'video/x-raw, format=BGR ! '
                'appsink').format(uri, latency, width, height)
    else:
        raise RuntimeError('H.264 or H.265 decoder not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def stream_func(self, uri, width = 1920, height = 1080, stream_queue = multiprocessing.Queue(maxsize=2)):
    cap = rtsp_stream_decode(uri, width, height, latency=100)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cur_time = time.time()
            stream_data = [frame, cur_time]
            if stream_queue.full():
                _ = stream_queue.get()
            stream_queue.put(stream_queue)
        else:
            break
        
if __name__ == '__main__':
    uri = 'rtsp://admin:@192.168.1.155:554'
    width = 1920
    height = 1080
    stream_queue = multiprocessing.Queue(maxsize=2)

    thread_stream = threading.Thread(target=stream_func, args=(uri, width, height, stream_queue))
    thread_stream.start()

    while True:
        if stream_queue.empty():
            time.sleep(0.002)
            continue
        stream_data = stream_queue.get()
        frame = stream_data[0]
        cv2.imshow('res', frame)
        cv2.waitKey(0)
