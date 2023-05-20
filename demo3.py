import cv2 as cv
from pathlib import Path
import torch
from torchvision import transforms 
from torchaudio.transforms import MFCC
from model.faceDetector import S3FD
from model.Model import ASD_Model
from collections import OrderedDict, deque
from threading import Thread
import ffmpeg
import numpy as np

MAX_FRAMES = 10

class Video(Thread):
    
    def __init__(self):
        super().__init__()
        self._stream = None
        self._queue = deque([], maxlen=MAX_FRAMES)
        self.daemon = True
        self._width = 640
        self._height = 480
        self._convert = transforms.ToTensor()
        
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
    
    @property
    def tensor(self):
        if len(self.stream) != MAX_FRAMES:
            return None
        print("video")
        tmp = []
        video_data = torch.empty([MAX_FRAMES, 3, self._height, self._width])
        for x in self.stream.copy():
            tensor = self._convert(x[0])
            C, H, W = tensor.size()
            tensor = tensor.view([1, 3, H, W])
            tmp.append(tensor)
        video_data = torch.cat(tmp, dim=0)
        video_data = torch.swapaxes(video_data, 1, 3)
        return video_data

class Audio(Thread):
    
    def __init__(self):
        super().__init__()
        self._stream = None
        self._queue = deque([], maxlen=MAX_FRAMES)
        self.daemon = True
        self._sr = 16000
        self._channels = 1
        self._mfcc_convert = MFCC(n_mfcc=13)
    
    def run(self):
        out = (
        ffmpeg
        .input('hw:0', f="alsa")
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=f"{self._channels}", ar=f'{self._sr}')
        .run_async(pipe_stdout=True)
        )
        chunk_size = self._sr
        
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
    
    @property
    def tensor(self):
        if len(self.stream) != MAX_FRAMES:
            return None
        audio_data = torch.empty([MAX_FRAMES, 13, 80])
        tmp = []
        for x in self.stream.copy():
            x = self.pcm2float(x)
            x = torch.tensor(x, dtype=torch.float32)
            mfcc = self._mfcc_convert(x)
            tmp.append(mfcc)
        audio_data = torch.cat(tmp, dim=0)
        return audio_data
    
    def pcm2float(self, sig, dtype='float32'):
        """Convert PCM signal to floating point with a range from -1 to 1.
        Use dtype='float32' for single precision.
        Parameters
        ----------
        sig : array_like
            Input array, must have integral type.
        dtype : data type, optional
            Desired (floating point) data type.
        Returns
        -------
        numpy.ndarray
            Normalized floating point data.
        See Also
        --------
        float2pcm, dtype
        """
        sig = np.asarray(sig)
        if sig.dtype.kind not in 'iu':
            raise TypeError("'sig' must be an array of integers")
        dtype = np.dtype(dtype)
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(sig.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig.astype(dtype) - offset) / abs_max
    
if __name__ == "__main__":
    print("Cuda available:", torch.cuda.is_available())
    curr_dir = Path(__file__).parent
    face_detector = S3FD()
    
    model_path = curr_dir.joinpath("weight", "pretrain_AVA_CVPR.model")
    weights = torch.load(model_path)

    weights_nw = OrderedDict()
    for k in weights.keys():
        try:
            k_nw = k.split("model.")[1]
            weights_nw[k_nw] = weights[k]
        except IndexError:
            continue
    
    del weights

    model = ASD_Model()
    model.load_state_dict(weights_nw)
    model.eval()
    
    video = Video()
    audio = Audio()
    
    pil_conv = transforms.ToPILImage()
    
    video.start()
    audio.start()
    
    while True:
        try:
            video_data = video.tensor
            audio_data = audio.tensor
            if video_data is None: continue
        except IndexError:
            continue
        img = video_data[-1]
        img = torch.swapaxes(img, 0, 2)
        img = pil_conv(img)
        img = np.array(img)
        #img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        bboxes = face_detector.detect_faces(img)
        for bbox in bboxes:
            x, y, w, h, _ = bbox
            cv.rectangle(img, [int(x), int(y)], [int(w), int(h)], (0, 0, 255), 3)
        #print(img.shape)
        cv.imshow("test", img)
        cv.waitKey(1)