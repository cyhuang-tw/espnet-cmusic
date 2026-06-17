import os, json, statistics as st
# SE_ROOT = working root with eval/scores/<system>/score/<metric>.scp (set in env).
SE=os.environ["SE_ROOT"]
SYS=["noisy","fullft","lora"]
SCALAR=[("dnsmos/DNSMOS_OVRL","DNSMOS↑"),
        ("nisqa/NISQA_MOS","NISQA↑"),("utmos/UTMOS","UTMOS↑"),("scoreq/Scoreq","SCOREQ↑"),
        ("speechbert/SpeechBERTScore","SpBERT↑"),("lps/PhonemeSimilarity","PhonSim↑"),
        ("spk_sim/SpeakerSimilarity","SpkSim↑"),("lid/LAcc","LID-Acc↑")]
EDIT=[("wer/CER","CER%↓"),("wer/WER","WER%↓")]
def scalar_mean(p):
    if not os.path.exists(p): return None,0
    v=[]
    for ln in open(p):
        x=ln.split()
        if len(x)>=2:
            try:
                f=float(x[1])
                if f==f: v.append(f)
            except: pass
    return (sum(v)/len(v) if v else None), len(v)
def edit_rate(p):  # corpus-level: sum errors / sum ref_len * 100
    if not os.path.exists(p): return None,0
    err=ref=n=0
    for ln in open(p):
        try:
            d=json.loads(ln.split(None,1)[1])
            err+=d["delete"]+d["insert"]+d["replace"]; ref+=d["delete"]+d["replace"]+d["equal"]; n+=1
        except: pass
    return (100*err/ref if ref else None), n
def row(lab,fn,mf):
    cells=[]
    for s in SYS:
        m,nn=fn(f"{SE}/eval/scores/{s}/score/{mf}.scp")
        cells.append(f"{m:.3f}({nn})" if m is not None else "   —")
    return lab,cells
print(f"{'metric':<11}"+"".join(f"{s:>15}" for s in SYS))
for mf,lab in SCALAR: 
    l,c=row(lab,scalar_mean,mf); print(f"{l:<11}"+"".join(f"{x:>15}" for x in c))
for mf,lab in EDIT:
    l,c=row(lab,edit_rate,mf); print(f"{l:<11}"+"".join(f"{x:>15}" for x in c))
