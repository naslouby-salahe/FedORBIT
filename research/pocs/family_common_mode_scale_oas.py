"""Hierarchical common-mode scale update across N-BaIoT decay windows."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys,zlib,re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"research"/"pocs"),str(ROOT/"src")]
from dependence_adjusted_oas import mean_ess
from fedorbit.detection.gaussian import scorer_from_rows
from fedorbit.detection.moments import aggregate_equal_weight,summarise

DATA=ROOT/"outputs"/"prepared"/"nbaiot"; RAW=ROOT/"data"/"raw"/"N-BaIoT"
OUT=ROOT/"research"/"pocs"/"results"/"family_common_mode_scale_oas.csv"
DEVICES=("Danmini_Doorbell","Ecobee_Thermostat","Ennio_Doorbell","Philips_B120N10_Baby_Monitor",
"Provision_PT_737E_Security_Camera","Provision_PT_838_Security_Camera","Samsung_SNH_1011_N_Webcam",
"SimpleHome_XCS7_1002_WHT_Security_Camera","SimpleHome_XCS7_1003_WHT_Security_Camera")
WINDOWS=("L5","L3","L1","L0.1","L0.01"); SUPPORTS=(30,100,300,1000)
REPS=30; SEED=202609313; RATE=.01; EVAL_N=5000; FLOOR=1e-8


def feature_groups():
    columns=pd.read_csv(RAW/DEVICES[0]/"benign_traffic.csv",nrows=0).columns.tolist()
    grouped={}; pattern=re.compile(r"(.+?)_(L5|L3|L1|L0\.1|L0\.01)_(.+)$")
    for i,name in enumerate(columns):
        m=pattern.fullmatch(name)
        if m is None: raise ValueError(name)
        prefix,window,stat=m.groups(); grouped.setdefault(f"{prefix}::{stat}",{})[window]=i
    if len(grouped)!=23 or any(set(x)!=set(WINDOWS) for x in grouped.values()): raise ValueError("schema")
    return [np.array([grouped[k][w] for w in WINDOWS],dtype=int) for k in sorted(grouped)]


def peer_profile(peer_summaries,groups):
    centers=np.stack([s.mean for s in peer_summaries])
    within=np.stack([s.covariance.diagonal() for s in peer_summaries])
    total=np.maximum(within+(centers-centers.mean(axis=0))**2,FLOOR)
    profiles=np.stack([np.stack([np.log(total[k,idx]) for idx in groups]) for k in range(len(centers))])
    # Match the accepted arithmetic predictive-marginal scale exactly at
    # zero local borrowing; only the common-mode update is novel here.
    center=np.stack([np.log(np.mean(total[:,idx],axis=0)) for idx in groups])
    # Between-peer variance of the common five-window shift, after subtracting
    # the estimated peer feature residual profile.
    residual=profiles-center[None,:,:]
    common=np.mean(residual,axis=2)
    tau2=np.maximum(np.var(common,axis=0,ddof=1),0.025**2)
    profile_var=np.maximum(np.var(residual-common[:,:,None],axis=0,ddof=1),0.01**2)
    return center,tau2,profile_var


def adapted_scale(rows,groups,peer_profile_center,tau2,peer_profile_var):
    centered=rows-rows.mean(axis=0,keepdims=True)
    local_var=np.maximum(np.mean(centered**2,axis=0),FLOOR)
    ess=np.maximum(mean_ess(centered**2),2.0)
    df=np.maximum(ess-1,1.0)
    log_local=np.log(local_var)-(digamma(df/2)-np.log(df/2))
    obs_var=2.0/df
    scale=np.empty(rows.shape[1]); shift_weights=[]
    for g,idx in enumerate(groups):
        # Independent-window precision combines only the common family shift;
        # peer profile uncertainty remains feature-specific.
        residual=log_local[idx]-peer_profile_center[g]
        total_var=obs_var[idx]+peer_profile_var[g]
        precision=1.0/np.maximum(total_var,FLOOR)
        q=1.0/np.sum(precision)
        weight=tau2[g]/(tau2[g]+q)
        delta=weight*np.sum(precision*residual)/np.sum(precision)
        scale[idx]=np.exp(0.5*(peer_profile_center[g]+delta))
        shift_weights.append(weight)
    return scale,float(np.median(shift_weights))


def fit(rows,peer,groups,pc,tau2,pv,method):
    if method=="shared_oas": return scorer_from_rows(rows,peer.mean,peer.standard_deviation+1e-6)
    if method=="c104_peer_scale": return scorer_from_rows(rows,rows.mean(axis=0),peer.standard_deviation+1e-6)
    scale,_=adapted_scale(rows,groups,pc,tau2,pv)
    return scorer_from_rows(rows,rows.mean(axis=0),scale+1e-6)


def crossfit(rows,peer,groups,pc,tau2,pv,method):
    mid=len(rows)//2;a,b=rows[:mid],rows[mid:]
    return np.r_[fit(b,peer,groups,pc,tau2,pv,method).score(a),fit(a,peer,groups,pc,tau2,pv,method).score(b)]


def main():
    groups=feature_groups(); devices={x:np.load(DATA/f"{x}.npz") for x in DEVICES}; rec=[]
    summaries={x:summarise(devices[x]["support_pool"]) for x in DEVICES}
    for target in DEVICES:
        peers=[x for x in DEVICES if x!=target]; ps=[summaries[x] for x in peers]
        peer=aggregate_equal_weight(ps); pc,tau2,pv=peer_profile(ps,groups); pool=devices[target]["support_pool"]
        for n in SUPPORTS:
            for rep in range(REPS):
                rng=np.random.default_rng(SEED ^ zlib.crc32(f"{target}:{n}:{rep}".encode()))
                start=int(rng.integers(0,len(pool)-n+1));rows=pool[start:start+n]
                scale,ww=adapted_scale(rows,groups,pc,tau2,pv)
                benign,attack=devices[target]["test_benign"],devices[target]["test_attack"]
                if len(benign)>EVAL_N:benign=benign[rng.choice(len(benign),EVAL_N,replace=False)]
                if len(attack)>EVAL_N:attack=attack[rng.choice(len(attack),EVAL_N,replace=False)]
                for method in ("shared_oas","c104_peer_scale","family_common_mode"):
                    model=fit(rows,peer,groups,pc,tau2,pv,method)
                    threshold=float(np.quantile(crossfit(rows,peer,groups,pc,tau2,pv,method),1-RATE))
                    sb,sa=model.score(benign),model.score(attack); y=np.r_[np.zeros(len(sb)),np.ones(len(sa))];s=np.r_[sb,sa]
                    rec.append({"device":target,"support_n":n,"replicate":rep,"offset":start,"method":method,
                     "auroc":roc_auc_score(y,s),"spauc01":roc_auc_score(y,s,max_fpr=RATE),"fpr":np.mean(sb>threshold),
                     "tpr":np.mean(sa>threshold),"median_group_local_weight":ww})
            print(target,n,"done",flush=True)
    frame=pd.DataFrame(rec);OUT.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(OUT,index=False)
    by=frame.groupby(["support_n","method","device"])[["auroc","spauc01","fpr","tpr","median_group_local_weight"]].mean().reset_index()
    print(by.groupby(["support_n","method"]).mean(numeric_only=True).round(4).to_string());print("wrote",len(frame),OUT)


if __name__=="__main__":main()
