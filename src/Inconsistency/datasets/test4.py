import torchaudio
waveframe,_=torchaudio.load("/workspace/datasets/DAICWOZ/300_P/300_AUDIO.wav")
print("*"*30)
print(waveframe)