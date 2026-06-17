import sys, os, soundfile as sf, numpy as np
inf_scp, ref_scp, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir+"/inf", exist_ok=True); os.makedirs(outdir+"/ref", exist_ok=True)
inf={l.split()[0]:l.split()[1] for l in open(inf_scp)}
ref={l.split()[0]:l.split()[1] for l in open(ref_scp)}
fi=open(outdir+"/inf_al.scp","w"); fr=open(outdir+"/ref_al.scp","w")
n=0
for uid in inf:
    if uid not in ref: continue
    ai,si=sf.read(inf[uid]); ar,sr=sf.read(ref[uid])
    if ai.ndim>1: ai=ai[:,0]
    if ar.ndim>1: ar=ar[:,0]
    m=min(len(ai),len(ar))
    ip=f"{outdir}/inf/{uid}.wav"; rp=f"{outdir}/ref/{uid}.wav"
    sf.write(ip,ai[:m].astype('float32'),si,subtype='PCM_16')
    sf.write(rp,ar[:m].astype('float32'),sr,subtype='PCM_16')
    fi.write(f"{uid} {ip}\n"); fr.write(f"{uid} {rp}\n"); n+=1
fi.close(); fr.close(); print(f"aligned {n} pairs -> {outdir}")
