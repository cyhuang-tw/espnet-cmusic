import os, json
# SE_ROOT = working root with eval/scores/<system>/score/; URGENT_DATA = URGENT challenge dir.
SE=os.environ["SE_ROOT"]
R=os.environ["URGENT_DATA"]
SYS=["noisy","fullft","lora"]
cat={}
for ln in open(f"{R}/data/validation/utt2category"):
    u,c=ln.split(); cat[u]=c
def load_scalar(s,mf):
    d={}
    p=f"{SE}/eval/scores/{s}/score/{mf}.scp"
    if os.path.exists(p):
        for ln in open(p):
            x=ln.split()
            if len(x)>=2:
                try: d[x[0]]=float(x[1])
                except: pass
    return d
def load_wer(s):  # per-uid (errors, reflen)
    d={}
    p=f"{SE}/eval/scores/{s}/score/wer/WER.scp"
    if os.path.exists(p):
        for ln in open(p):
            try:
                u=ln.split(None,1)[0]; j=json.loads(ln.split(None,1)[1])
                d[u]=(j["delete"]+j["insert"]+j["replace"], j["delete"]+j["replace"]+j["equal"])
            except: pass
    return d
dns={s:load_scalar(s,"dnsmos/DNSMOS_OVRL") for s in SYS}
wer={s:load_wer(s) for s in SYS}
# per-category DNSMOS + WER
cats=sorted(set(cat.values()))
print("=== PER-CATEGORY: DNSMOS (↑) | WER% (↓) ===")
print(f"{'category':<16}"+"".join(f"{s:>20}" for s in SYS))
for c in cats:
    us=[u for u in cat if cat[u]==c]
    row=[]
    for s in SYS:
        dv=[dns[s][u] for u in us if u in dns[s]]
        we=sum(wer[s][u][0] for u in us if u in wer[s]); wr=sum(wer[s][u][1] for u in us if u in wer[s])
        dm=sum(dv)/len(dv) if dv else 0; wm=100*we/wr if wr else 0
        row.append(f"{dm:.2f}|{wm:.1f}%")
    print(f"{c:<16}"+"".join(f"{x:>20}" for x in row))
# per-clip: fraction where enhanced WER > noisy WER
print("\n=== Per-clip: fraction of clips where enhancement makes ASR WORSE than noisy ===")
def perclip_wer(s,u):
    if u in wer[s] and wer[s][u][1]>0: return wer[s][u][0]/wer[s][u][1]
    return None
common=[u for u in wer["noisy"] if u in wer["fullft"] and u in wer["lora"]]
for s in ["fullft","lora"]:
    worse=better=same=0
    for u in common:
        n=perclip_wer("noisy",u); e=perclip_wer(s,u)
        if n is None or e is None: continue
        if e>n+1e-9: worse+=1
        elif e<n-1e-9: better+=1
        else: same+=1
    tot=worse+better+same
    print(f"{s}: WORSE than noisy on {worse}/{tot} ({100*worse/tot:.0f}%), better {better} ({100*better/tot:.0f}%), same {same}")
