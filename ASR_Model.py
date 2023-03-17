import torch
from torch import nn
from transformers import GPT2TokenizerFast
import model2
import whisper


class Model_Script_Asr(nn.Module):
    def __init__(self,path_script,path_token):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = path_script
        self.path_token = path_token
        self.model = torch.jit.load(self.path).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.path_token)
        
    def forward(self,mel_input):
        tokens = self.tokenizer.encode("<|startoftranscript|><|en|><|transcribe|><|notimestamps|>", return_tensors="pt").to(self.device)
        mel = mel_input.to(self.device)
        suppress_blanks = [220, 50257]
        suppress_nonspeech = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 
                                93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 
                                3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 
                                14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 
                                32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362]
        with torch.no_grad():    
            scripted_transcribed = self.model.greedy_decode(tokens,mel,suppress_blanks,suppress_nonspeech)
            text = self.tokenizer.batch_decode(scripted_transcribed,skip_special_tokens=True)[0]
        return text
    
    
if __name__ == '__main__':
    model_path = './checkpoint/tiny_asr.pth'
    token_path = './whisper/assets/whisper_mult_gpt2'
    model = Model_Script_Asr(model_path,token_path)
    audiolist = [
        "tests/jfk.flac",
        "tests/jfk_noise_front.wav",
        "tests/jfk_noise_middle.wav",
        "tests/jfk_noise_back.wav",
        "tests/noise_only.wav",
        "tests/debussy.wav",
    ]
    audio = whisper.load_audio(audiolist[2])
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)
    
    text = model(mel)
    print(text)