"""Block-instability uncertainty for continuous peer/local scale adaptation."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"research"/"pocs"),str(ROOT/"src")]
from dependence_adjusted_oas import mean_ess
from mismatch_safe_scale_oas import peer_prior
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight,summarise

DATA=ROOT/"outputs"/"prepared"/"nbaiot"
OUT=ROOT/"research"/"pocs"/"results"/"block_stability_scale_oas_nbaiot.csv"
DEVICES=("Danmini_Doorbell","Ecobee_Thermostat","Ennio_Doorbell",
 "Philips_B120N10_Baby_Monitor","Provision_PT_737E_Security_Camera",
 "Provision_PT_838_Security_Camera","Samsung_SNH_1011_N_Webcam",
 "SimpleHome_XCS7_1002_WHT_Security_Camera","SimpleHome_XCS7_1003_WHT_Security_Camera")
SUPPORTS=(30,100,300,1000); REPS=30; SEED=202609313; RATE=.01; EVAL_N=5000; FLOOR=1e-8


def block_posterior_scale(rows, peer_scale, tau2, block_count=3):
    centered=rows-rows.mean(axis=0,keepdims=True)
    local_var=np.maximum(np.mean(centered**2,axis=0),FLOOR)
    ess=np.maximum(mean_ess(centered**2),2.0)
    df=np.maximum(ess-1,1.0)
    bias=digamma(df/2)-np.log(df/2)
    log_local=np.log(local_var)-bias
    base_noise=2.0/df
    block_logs=[]; block_noise=[]
    for block in np.array_split(rows,block_count):
        bc=block-block.mean(axis=0,keepdims=True)
        bv=np.maximum(np.mean(bc**2,axis=0),FLOOR)
        be=np.maximum(mean_ess(bc**2),2.0)
        bd=np.maximum(be-1.0,1.0)
        block_logs.append(np.log(bv)-(digamma(bd/2)-np.log(bd/2)))
        block_noise.append(2.0/bd)
    observed=np.var(np.stack(block_logs),axis=0,ddof=1)
    expected=np.mean(np.stack(block_noise),axis=0)
    excess=np.maximum(observed-expected,0.0)
    observation_var=base_noise+excess
    weight=tau2/(tau2+observation_var+FLOOR)
    log_prior=np.log(np.maximum(peer_scale**2,FLOOR))
    posterior=log_prior+weight*(log_local-log_prior)
    return np.exp(0.5*posterior),weight,ess,excess


def fit(rows,peer,prior_scale,tau2,method):
    if method=="shared_oas":
        return scorer_from_rows(rows,peer.mean,peer.standard_deviation+1e-6)
    if method=="peer_scale_local_center":
        return scorer_from_rows(rows,rows.mean(axis=0),peer.standard_deviation+1e-6)
    if method=="scale_eb_local_center":
        scale,_,_=__import__("mismatch_safe_scale_oas").posterior_scale(rows,prior_scale,tau2)
    else:
        scale,_,_,_=block_posterior_scale(rows,prior_scale,tau2)
    return scorer_from_rows(rows,rows.mean(axis=0),scale+1e-6)


def crossfit(rows,peer,scale,tau2,method):
    mid=len(rows)//2; a,b=rows[:mid],rows[mid:]
    return np.r_[fit(b,peer,scale,tau2,method).score(a),fit(a,peer,scale,tau2,method).score(b)]


def main():
    devices={name:np.load(DATA/f"{name}.npz") for name in DEVICES}; rec=[]
    for target in DEVICES:
        peer_names=[x for x in DEVICES if x!=target]
        summaries=[summarise(devices[x]["support_pool"]) for x in peer_names]
        peer=aggregate_equal_weight(summaries)
        _,prior_scale,tau2,_=peer_prior(summaries)
        pool=devices[target]["support_pool"]
        for n in SUPPORTS:
            for rep in range(REPS):
                rng=np.random.default_rng(SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode()))
                start=int(rng.integers(0,len(pool)-n+1)); rows=pool[start:start+n]
                benign,attack=devices[target]["test_benign"],devices[target]["test_attack"]
                if len(benign)>EVAL_N: benign=benign[rng.choice(len(benign),EVAL_N,replace=False)]
                if len(attack)>EVAL_N: attack=attack[rng.choice(len(attack),EVAL_N,replace=False)]
                block_scale,w,ess,excess=block_posterior_scale(rows,prior_scale,tau2)
                for method in ("shared_oas","peer_scale_local_center","scale_eb_local_center","block_scale_local_center"):
                    model=fit(rows,peer,prior_scale,tau2,method)
                    threshold=float(np.quantile(crossfit(rows,peer,prior_scale,tau2,method),1-RATE))
                    sb,sa=model.score(benign),model.score(attack)
                    y=np.r_[np.zeros(len(sb)),np.ones(len(sa))]; s=np.r_[sb,sa]
                    rec.append({"device":target,"support_n":n,"replicate":rep,"offset":start,
                     "method":method,"auroc":roc_auc_score(y,s),"spauc01":roc_auc_score(y,s,max_fpr=RATE),
                     "fpr":np.mean(sb>threshold),"tpr":np.mean(sa>threshold),
                     "median_scale_local_weight":np.median(w),"median_scale_ess":np.median(ess),
                     "median_block_excess":np.median(excess)})
            print(target,n,"done",flush=True)
    frame=pd.DataFrame(rec);OUT.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(OUT,index=False)
    dev=frame.groupby(["support_n","method","device"])[["auroc","spauc01","fpr","tpr","median_scale_local_weight","median_block_excess"]].mean().reset_index()
    print(dev.groupby(["support_n","method"]).mean(numeric_only=True).round(4).to_string())
    print("wrote",len(frame),OUT)


if __name__=="__main__":main()
