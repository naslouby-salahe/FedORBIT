"""Full peer-predictive covariance target with risk-derived local shrinkage."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
import sys,zlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"research"/"pocs"))
from fedorbit.detection.gaussian import GaussianScorer,scorer_from_rows,shrunk_precision
from fedorbit.detection.moments import aggregate_equal_weight,summarise
from dependence_adjusted_oas import covariance_ess
DATA=ROOT/"outputs"/"prepared"/"nbaiot";OUT=ROOT/"research"/"pocs"/"results"/"full_peer_covariance_target_nbaiot.csv"
DEVICES=("Danmini_Doorbell","Ecobee_Thermostat","Ennio_Doorbell","Philips_B120N10_Baby_Monitor","Provision_PT_737E_Security_Camera","Provision_PT_838_Security_Camera","Samsung_SNH_1011_N_Webcam","SimpleHome_XCS7_1002_WHT_Security_Camera","SimpleHome_XCS7_1003_WHT_Security_Camera")
SUPPORTS=(30,100,300,1000);REPS=30;SEED=202609313;RATE=.01;EVAL_N=5000;FLOOR=1e-9

def model(rows,peer,peer_summaries,method):
    scale=peer.standard_deviation+1e-6; center=rows.mean(axis=0)
    if method=="shared_oas":return scorer_from_rows(rows,peer.mean,scale),1.0
    local=(rows-center)/scale; S=local.T@local/len(rows)
    if method=="local_oas":return scorer_from_rows(rows,center,scale),0.0
    # Full predictive peer covariance combines within-client covariance and
    # covariance of client centers, matching the peer marginal estimand.
    peer_cov=peer.covariance/np.outer(scale,scale)
    peer_cov=(peer_cov+peer_cov.T)/2
    if method=="peer_cov_fixed":
        C=peer_cov
        alpha=1.0
    elif method in ("peer_cov_soft","risk_soft_cov","risk_soft_cov_mismatch"):
        stack=[]; peer_est_risk=[]
        for summary in peer_summaries:
            delta=(summary.mean-peer.mean)/scale
            within=summary.covariance/np.outer(scale,scale)
            ck=(within+np.outer(delta,delta)); stack.append(ck)
            trk=float(np.trace(ck)); sqk=float(np.sum(ck*ck))
            peer_est_risk.append((trk*trk+sqk)/max(float(summary.weight)-1.0,1.0))
        stack=np.stack(stack)
        local_ess=covariance_ess(local)
        tr=float(np.trace(S)); trsq=float(np.sum(S*S))
        local_risk=float((tr*tr+trsq)/max(local_ess-1.0,1.0))
        logw=-0.5*np.sum((stack-S[None,:,:])**2,axis=(1,2))/np.maximum(local_risk+np.asarray(peer_est_risk),FLOOR)
        logw-=np.max(logw); weights=np.exp(logw); weights/=weights.sum()
        C=np.einsum("k,kij->ij",weights,stack)
        peer_risk=float(np.sum(weights**2*np.asarray(peer_est_risk)))
        target_mismatch=0.0
        if method=="risk_soft_cov_mismatch":
            target_mismatch=max(float(np.sum((S-C)**2))-local_risk-peer_risk,FLOOR)
        alpha=1.0 if method=="peer_cov_soft" else local_risk/max(local_risk+peer_risk+target_mismatch,FLOOR)
        if method!="peer_cov_soft": C=(1-alpha)*S+alpha*C
    else:
        matrices=[]
        for summary in peer_summaries:
            delta=(summary.mean-peer.mean)/scale
            within=summary.covariance/np.outer(scale,scale)
            matrices.append(within+np.outer(delta,delta))
        stack=np.stack(matrices); avg=np.mean(stack,axis=0)
        peer_risk=float(np.sum((stack-avg[None,:,:])**2)/(len(stack)*(len(stack)-1)))
        tr=float(np.trace(S)); trsq=float(np.sum(S*S))
        # Gaussian-Wishart first-order estimate of Frobenius variance for the
        # local covariance, with a conservative AR-free nominal support here.
        local_ess=covariance_ess(local)
        local_risk=float((tr*tr+trsq)/max(local_ess-1,1))
        target_mismatch=0.0
        if method=="risk_blend_mismatch":
            # Estimate target-specific peer-target bias by debiasing the
            # observed Frobenius discrepancy for local and peer estimator risk.
            target_mismatch=max(float(np.sum((S-peer_cov)**2))-local_risk-peer_risk,0.0)
        alpha=local_risk/max(local_risk+peer_risk+target_mismatch,FLOOR)
        C=(1-alpha)*S+alpha*peer_cov
    C=(C+C.T)/2
    precision=shrunk_precision(C,float(len(rows)))
    return GaussianScorer(center,scale,precision),alpha

def crossfit(rows,peer,ps,method):
    mid=len(rows)//2;a,b=rows[:mid],rows[mid:]
    return np.r_[model(b,peer,ps,method)[0].score(a),model(a,peer,ps,method)[0].score(b)]

def main():
    devices={k:np.load(DATA/f"{k}.npz") for k in DEVICES};rec=[]
    for target in DEVICES:
        names=[k for k in DEVICES if k!=target];ps=[summarise(devices[k]["support_pool"]) for k in names]
        peer=aggregate_equal_weight(ps);pool=devices[target]["support_pool"]
        for n in SUPPORTS:
            for rep in range(REPS):
                rng=np.random.default_rng(SEED^zlib.crc32(f"{target}:{n}:{rep}".encode()));start=int(rng.integers(0,len(pool)-n+1));rows=pool[start:start+n]
                benign,attack=devices[target]["test_benign"],devices[target]["test_attack"]
                if len(benign)>EVAL_N:benign=benign[rng.choice(len(benign),EVAL_N,replace=False)]
                if len(attack)>EVAL_N:attack=attack[rng.choice(len(attack),EVAL_N,replace=False)]
                for method in ("shared_oas","local_oas","peer_cov_fixed","risk_blend","risk_blend_mismatch",
                               "peer_cov_soft","risk_soft_cov","risk_soft_cov_mismatch"):
                    fitted,alpha=model(rows,peer,ps,method);threshold=float(np.quantile(crossfit(rows,peer,ps,method),1-RATE));sb,sa=fitted.score(benign),fitted.score(attack)
                    y=np.r_[np.zeros(len(sb)),np.ones(len(sa))];s=np.r_[sb,sa]
                    rec.append({"device":target,"support_n":n,"replicate":rep,"offset":start,"method":method,"peer_weight":alpha,
                      "auroc":roc_auc_score(y,s),"spauc01":roc_auc_score(y,s,max_fpr=RATE),"fpr":np.mean(sb>threshold),"tpr":np.mean(sa>threshold)})
            print(target,n,"done",flush=True)
    frame=pd.DataFrame(rec);OUT.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(OUT,index=False)
    by=frame.groupby(["support_n","method","device"])[["auroc","spauc01","fpr","tpr","peer_weight"]].mean().reset_index()
    print(by.groupby(["support_n","method"]).mean(numeric_only=True).round(4).to_string());print("wrote",len(frame),OUT)

if __name__=="__main__":main()
