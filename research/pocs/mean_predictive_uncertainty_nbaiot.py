"""Integrate posterior uncertainty in the benign target mean into score scale."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
import sys,zlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/"research"/"pocs"),str(ROOT/"src")]
from dependence_adjusted_oas import mean_ess
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight,summarise
DATA=ROOT/"outputs"/"prepared"/"nbaiot";OUT=ROOT/"research"/"pocs"/"results"/"mean_predictive_uncertainty_nbaiot.csv"
DEVICES=("Danmini_Doorbell","Ecobee_Thermostat","Ennio_Doorbell","Philips_B120N10_Baby_Monitor","Provision_PT_737E_Security_Camera","Provision_PT_838_Security_Camera","Samsung_SNH_1011_N_Webcam","SimpleHome_XCS7_1002_WHT_Security_Camera","SimpleHome_XCS7_1003_WHT_Security_Camera")
SUPPORTS=(30,100,300,1000);REPS=30;SEED=202609313;RATE=.01;EVAL_N=5000;FLOOR=1e-9

def mean_update(rows,peer,peer_var):
    local=rows.mean(0); local_var=np.maximum(rows.var(0,ddof=1),FLOOR); ess=np.maximum(mean_ess(rows),2.0)
    sample_var=local_var/ess; post_var=1/(1/np.maximum(peer_var,FLOOR)+1/np.maximum(sample_var,FLOOR))
    weight=peer_var/(peer_var+sample_var+FLOOR)
    return peer.mean+weight*(local-peer.mean),post_var

def fit(rows,peer,peer_var,method):
    if method=="shared_oas":return scorer_from_rows(rows,peer.mean,peer.standard_deviation+1e-6)
    local=rows.mean(0)
    if method=="local_peer_scale":return scorer_from_rows(rows,local,peer.standard_deviation+1e-6)
    center,post_var=mean_update(rows,peer,peer_var)
    scale=peer.standard_deviation
    if method=="posterior_mean_only":pass
    elif method=="posterior_mean_predictive":scale=np.sqrt(scale**2+post_var)
    else:raise ValueError(method)
    return scorer_from_rows(rows,center,scale+1e-6)

def crossfit(rows,peer,pv,method):
    mid=len(rows)//2;a,b=rows[:mid],rows[mid:]
    return np.r_[fit(b,peer,pv,method).score(a),fit(a,peer,pv,method).score(b)]

def main():
    devices={k:np.load(DATA/f"{k}.npz") for k in DEVICES};rec=[]
    for target in DEVICES:
        peers=[k for k in DEVICES if k!=target];ss=[summarise(devices[k]["support_pool"]) for k in peers]
        peer=aggregate_equal_weight(ss);means=np.stack([s.mean for s in ss]);pv=np.maximum(np.var(means,axis=0,ddof=1),FLOOR);pool=devices[target]["support_pool"]
        for n in SUPPORTS:
            for rep in range(REPS):
                rng=np.random.default_rng(SEED^zlib.crc32(f"{target}:{n}:{rep}".encode()));start=int(rng.integers(0,len(pool)-n+1));rows=pool[start:start+n]
                benign,attack=devices[target]["test_benign"],devices[target]["test_attack"]
                if len(benign)>EVAL_N:benign=benign[rng.choice(len(benign),EVAL_N,replace=False)]
                if len(attack)>EVAL_N:attack=attack[rng.choice(len(attack),EVAL_N,replace=False)]
                for method in ("shared_oas","local_peer_scale","posterior_mean_only","posterior_mean_predictive"):
                    model=fit(rows,peer,pv,method);threshold=float(np.quantile(crossfit(rows,peer,pv,method),1-RATE));sb,sa=model.score(benign),model.score(attack)
                    y=np.r_[np.zeros(len(sb)),np.ones(len(sa))];s=np.r_[sb,sa]
                    rec.append({"device":target,"support_n":n,"replicate":rep,"offset":start,"method":method,
                     "auroc":roc_auc_score(y,s),"spauc01":roc_auc_score(y,s,max_fpr=RATE),"fpr":np.mean(sb>threshold),"tpr":np.mean(sa>threshold)})
            print(target,n,"done",flush=True)
    frame=pd.DataFrame(rec);OUT.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(OUT,index=False)
    by=frame.groupby(["support_n","method","device"])[["auroc","spauc01","fpr","tpr"]].mean().reset_index()
    print(by.groupby(["support_n","method"]).mean(numeric_only=True).round(4).to_string());print("wrote",len(frame),OUT)

if __name__=="__main__":main()
