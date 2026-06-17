"""Resample the URGENT validation clean+noisy references to 16 kHz for metric scoring.
Paths come from the environment (cluster-agnostic):
  URGENT_DATA  - URGENT challenge dir (contains data/validation/{wav,spk1}.scp).
  SE_ROOT      - working root holding data/se_valid/dataset.json; refs written under $SE_ROOT/eval/refs/.
"""
import json, os, soundfile as sf, librosa, numpy as np
R=os.environ["URGENT_DATA"]
SE=os.environ["SE_ROOT"]
meta=json.load(open(SE+"/data/se_valid/dataset.json"))
uids=meta["samples"]
# resolve noisy/clean lhotse recordings -> per-uid source paths via wav.scp + spk1.scp
def loadscp(p):
    d={}
    for ln in open(p):
        k,v=ln.strip().split(None,1); d[k]=v
    return d
noisy=loadscp(R+"/data/validation/wav.scp")
clean=loadscp(R+"/data/validation/spk1.scp")
os.makedirs(SE+"/eval/refs/clean16k",exist_ok=True)
os.makedirs(SE+"/eval/refs/noisy16k",exist_ok=True)
fr=open(SE+"/eval/refs/ref16k.scp","w"); fn=open(SE+"/eval/refs/noisy16k.scp","w")
def to16k(p):
    p=p if p.startswith("/") else os.path.join(R,p)
    a,sr=sf.read(p)
    if a.ndim>1: a=a[:,0]
    if sr!=16000: a=librosa.resample(a.astype('float32'),orig_sr=sr,target_sr=16000)
    return a.astype('float32')
n=0
for u in uids:
    if u not in clean or u not in noisy: continue
    cp=f"{SE}/eval/refs/clean16k/{u}.wav"; npth=f"{SE}/eval/refs/noisy16k/{u}.wav"
    sf.write(cp,to16k(clean[u]),16000,subtype='PCM_16'); fr.write(f"{u} {cp}\n")
    sf.write(npth,to16k(noisy[u]),16000,subtype='PCM_16'); fn.write(f"{u} {npth}\n")
    n+=1
    if n%200==0: print(f"  {n}/{len(uids)}",flush=True)
fr.close(); fn.close(); print("DONE refs16k+noisy16k:",n)
