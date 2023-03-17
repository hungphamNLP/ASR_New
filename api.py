from fastapi import FastAPI, Body, Request, Form,UploadFile,File,Response,Headerfrom fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import whisper

from ASR_Model import Model_Script_Asr
import traceback
import torch
import sys
import os

from pathlib import Path
path_root = Path(__file__).parents[0]


app = FastAPI()

model_path = str(path_root) +'/checkpoint/tiny_asr.pth'
token_path = str(path_root)+'/whisper/assets/whisper_mult_gpt2'
model = Model_Script_Asr(model_path,token_path)

@app.post('/voice2text')
async def receiveData(request: Request,file: UploadFile= File(...)):
    audio_bytes = file.file.read()
    with open("audio.wav","wb") as f: 
        f.write(audio_bytes)
    
    audio = whisper.load_audio('audio.wav')
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)
    
    text = model(mel)
    return text