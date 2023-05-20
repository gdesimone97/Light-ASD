import cv2 as cv
from pathlib import Path
import torch
#from torchvision import transforms
from model.faceDetector import S3FD
#from model.Model import ASD_Model
from collections import OrderedDict, deque
from threading import Thread
import ffmpeg
import numpy as np

MAX_FRAMES = 30

class Video(Thread):
    
    def __init__(self):
        super().__init__()
        self._stream = None
        self._queue = deque([], maxlen=MAX_FRAMES)
        self.daemon = True
        self._width = 640
        self._height = 480
    
    def run(self):
        out = (
        ffmpeg
        .input('/dev/video0', s=f"{self._width}x{self._height}")
        .output('pipe:', format='rawvideo', pix_fmt="bgr24")
        .run_async(pipe_stdout=True)
        )
        while True:
            in_bytes = out.stdout.read(self._width * self._height * 3)
            if not in_bytes:
                break
            data = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([-1, self._height, self._width, 3])
            )
            self._queue.append(data)
        
    @property
    def stream(self):
        return self._queue

class Audio(Thread):
    
    def __init__(self):
        super().__init__()
        self._stream = None
        self._queue = deque([], maxlen=MAX_FRAMES)
        self.daemon = True
        self._sr = 16000
        self._channels = 1
    
    def run(self):
        bytes_per_sample = np.dtype(np.int16).itemsize
        frame_size = bytes_per_sample * self._channels
        out = (
        ffmpeg
        .input('hw:0', f="alsa")
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=f"{self._channels}", ar=f'{self._sr}')
        .run_async(pipe_stdout=True)
        )
        chunk_size = frame_size * self._sr
        
        while True:
            in_bytes = out.stdout.read(chunk_size)
            if not in_bytes:
                break
            data = (
                np
                .frombuffer(in_bytes, np.uint8)
            )
            self._queue.append(data)

        
    @property
    def stream(self):
        return self._queue
    
if __name__ == "__main__":
    print("Cuda available:", torch.cuda.is_available())
    curr_dir = Path(__file__).parent
    face_detector = S3FD()
    video = Video()
    audio = Audio()
    video.start()
    audio.start()
    
    while True:
        try:
            img = video.stream[-1][0]
        except IndexError:
            continue
        bboxes = face_detector.detect_faces(img)
        for bbox in bboxes:
            x, y, w, h, _ = bbox
            cv.rectangle(img, [int(x), int(y)], [int(w), int(h)], (0, 0, 255), 3)
        #print(img.shape)
        cv.imshow("test", img)
        cv.waitKey(1)