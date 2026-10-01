"""Direct two-target MTS comparator for collaborative covariance geometry."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
import sys,zlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/"research"/"pocs"),str(ROOT/"src")]
from dependence_adjusted_oas import covariance_ess
from fedorbit.detection.gaussian import GaussianScorer,scorer_from_rows,shrunk_precision
from fedorbit.detection.moments import aggregate_equal_weight,summarise
DATA=ROOT/"outputs"/"prepared"/"nbaiot";OUT=ROOT/"research"/"pocs"/"results"/"multitarget_covariance_shrinkage_nbaiot.csv"
DEVICES=("Danmini_Doorbell","Ecobee_Thermostat","Ennio_Doorbell","Philips_B120N10_Baby_Monitor","Provision_PT_737E_Security_Camera","Provision_PT_838_Security_Camera","Samsung_SNH_1011_N_Webcam","SimpleHome_XCS7_1002_WHT_Security_Camera","SimpleHome_XCS7_1003_WHT_Security_Camera")
SUPPORTS=(30,100,300,1000);REPS=30;SEED=202609313;RATE=.01;EVAL_N=5000;FLOOR=1e-9

def mt_weights(S,peer_cov,neff):
    p=len(S);T1=np.eye(p)*(np.trace(S)/p);T2=(peer_cov+peer_cov.T)/2
    D1=T1-S;D2=T2-S
    A=np.array([[np.sum(D1*D1),np.sum(D1*D2)],[np.sum(D1*D2),np.sum(D2*D2)]])
    var_entry=(np.outer(np.diag(S),np.diag(S))+S*S)/max(neff,1.0)
    b=float(np.sum(var_entry))
    def objective(a):return float(a@A@a-2*b*np.sum(a))
    res=minimize(objective,np.array([0.5,0.1]),method="SLSQP",bounds=((0,1),(0,1)),
                 constraints=({"type":"ineq","fun":lambda a:1-np.sum(a)},),
                 options={"ftol":1e-10,"maxiter":100})
    if not res.success:
        candidates=[np.zeros(2),np.array([1.,0.]),np.array([0.,1.])]
        for i in range(3):
            a=np.linspace(0,1,101)
            if i==0:candidates.extend(np.stack([a,np.zeros_like(a)],axis=1))
            elif i==1:candidates.extend(np.stack([np.zeros_like(a),a],axis=1))
            else:candidates.extend(np.stack([a,1-a],axis=1))
        return min(candidates,key=objective)
    return np.clip(res.x,0,1)

def model(rows,peer,peer_cov,peer_summaries,method):
    scale=peer.standard_deviation+1e-6;center=rows.mean(0)
    if method=="shared_oas":return scorer_from_rows(rows,peer.mean,scale),np.array([0.,0.])
    if method=="local_oas":return scorer_from_rows(rows,center,scale),np.array([0.,0.])
    z=(rows-center)/scale;S=z.T@z/len(z)
    if method in ("risk_blend","risk_blend_oas"):
        tr=np.trace(S);trsq=np.sum(S*S);neff=covariance_ess(z)
        local_risk=(tr*tr+trsq)/max(neff-1,1)
        peer_matrices=[]
        for summary in peer_summaries:
            delta=(summary.mean-peer.mean)/scale
            within=summary.covariance/np.outer(scale,scale)
            peer_matrices.append(within+np.outer(delta,delta))
        peer_stack=np.stack(peer_matrices);peer_avg=peer_stack.mean(axis=0)
        peer_risk=float(np.sum((peer_stack-peer_avg[None,:,:])**2)/(len(peer_stack)*(len(peer_stack)-1)))
        alpha=local_risk/max(local_risk+peer_risk,1e-9)
        C=(1-alpha)*S+alpha*peer_cov
        weights=np.array([0.,alpha])
    else:
        neff=covariance_ess(z);weights=mt_weights(S,peer_cov,neff)
        C=(1-weights.sum())*S+weights[0]*np.eye(len(S))*np.trace(S)/len(S)+weights[1]*peer_cov
    C=(C+C.T)/2+np.eye(len(S))*1e-8
    precision=shrunk_precision(C,float(len(rows))) if method in ("risk_blend_oas","mts_two_target_oas") else np.linalg.inv(C)
    return GaussianScorer(center,scale,precision),weights

def crossfit(rows,peer,pcov,peer_summaries,method):
    mid=len(rows)//2;a,b=rows[:mid],rows[mid:]
    return np.r_[model(b,peer,pcov,peer_summaries,method)[0].score(a),model(a,peer,pcov,peer_summaries,method)[0].score(b)]

def main():
    devices={k:np.load(DATA/f"{k}.npz") for k in DEVICES};rec=[]
    for target in DEVICES:
        peers=[k for k in DEVICES if k!=target];ps=[summarise(devices[k]["support_pool"]) for k in peers]
        peer=aggregate_equal_weight(ps);scale=peer.standard_deviation+1e-6;pool=devices[target]["support_pool"]
        pcov=peer.covariance/np.outer(scale,scale);pcov=(pcov+pcov.T)/2
        for n in SUPPORTS:
            for rep in range(REPS):
                rng=np.random.default_rng(SEED^zlib.crc32(f"{target}:{n}:{rep}".encode()));start=int(rng.integers(0,len(pool)-n+1));rows=pool[start:start+n]
                benign,attack=devices[target]["test_benign"],devices[target]["test_attack"]
                if len(benign)>EVAL_N:benign=benign[rng.choice(len(benign),EVAL_N,replace=False)]
                if len(attack)>EVAL_N:attack=attack[rng.choice(len(attack),EVAL_N,replace=False)]
                for method in ("shared_oas","local_oas","risk_blend","risk_blend_oas",
                               "mts_two_target","mts_two_target_oas"):
                    fitted,w=model(rows,peer,pcov,ps,method);threshold=float(np.quantile(crossfit(rows,peer,pcov,ps,method),1-RATE));sb,sa=fitted.score(benign),fitted.score(attack)
                    y=np.r_[np.zeros(len(sb)),np.ones(len(sa))];s=np.r_[sb,sa]
                    rec.append({"device":target,"support_n":n,"replicate":rep,"offset":start,"method":method,
                     "identity_weight":w[0],"peer_cov_weight":w[1],"auroc":roc_auc_score(y,s),"spauc01":roc_auc_score(y,s,max_fpr=RATE),
                     "fpr":np.mean(sb>threshold),"tpr":np.mean(sa>threshold)})
            print(target,n,"done",flush=True)
    frame=pd.DataFrame(rec);OUT.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(OUT,index=False)
    by=frame.groupby(["support_n","method","device"])[["auroc","spauc01","fpr","tpr","identity_weight","peer_cov_weight"]].mean().reset_index()
    print(by.groupby(["support_n","method"]).mean(numeric_only=True).round(4).to_string());print("wrote",len(frame),OUT)

if __name__=="__main__":main()
