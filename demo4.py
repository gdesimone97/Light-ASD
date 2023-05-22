import cv2 as cv
from pathlib import Path
import torch
import torchvision
from torchvision import transforms 
from torchaudio.transforms import MFCC
from model.faceDetector import S3FD
from model.Model import ASD_Model
from collections import OrderedDict, deque
from threading import Thread
import ffmpeg
import numpy as np
from ASD import ASD
from loss import lossAV
import torch.nn.functional as F
import python_speech_features
from scipy.io import wavfile

import warnings
warnings.filterwarnings("ignore")

FPS = 25

class MyLossAV(lossAV):
    def forward(self, x):
        x = x.squeeze(1)
        x = self.FC(x)
        predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
        return predLabel

def tensor_audio(audio, numFrames, fps):     
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * FPS / fps, winstep = 0.010 * FPS / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = np.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    audio = torch.FloatTensor(audio)
    return audio

def tensor_video(video: torch.Tensor):
    H = 112
    faces = []
    video_full = video.detach().numpy()
    print("Total:", video_full.shape[0])
    for i in range(video_full.shape[0]):
        frame = video_full[i]
        #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        bboxes = face_detector.detect_faces(frame)
        for bbox in bboxes:
            x, y, w, h, _ = bbox
            face = frame[int(y):int(h), int(x):int(w)]
            face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
            face = cv.resize(face, (H, H))
            face = torch.FloatTensor(face)
            face = face.unsqueeze(0)
            faces.append(face)
            print(f"\rStatus: {len(faces)}", end='')
    faces = torch.cat(faces, dim=0)
    print()
    return faces

def pcm2float(sig, dtype='float32'):
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

    asd = ASD()
    asd.loadParameters(str(model_path))
    #model.eval()
    s = asd.to("cuda")
    convert = torchvision.transforms.ToTensor()
    video_audio = torchvision.io.read_video("test.avi")
    video = video_audio[0]
    print("video:", video.shape) #T,H,W,C
    _, audio = wavfile.read("test.wav")
    print("audio:", audio.shape)
    fps = video_audio[2]["video_fps"]
    print("fps:", fps)
    audio = tensor_audio(audio[0], video.shape[0], fps)
    video = tensor_video(video)
    audio = torch.unsqueeze(audio, 0).cuda()
    video = torch.unsqueeze(video, 0).cuda()
    embedA = s.model.forward_audio_frontend(audio)
    embedV = s.model.forward_visual_frontend(video)	
    out = s.model.forward_audio_visual_backend(embedA, embedV)
    x = s.lossAV.FC(out)
    predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
    print(predLabel)