from __future__ import annotations

import argparse, json, math
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

A=(4,5,6); P=((4,5),(4,6),(5,6))

D=dict(
 dt=.002,t_end=12.,initial_prestructure_energy=12.,high_rank_capacity=4.8,
 formation_onset=.45,formation_offset=3.10,formation_width=.28,k_inflow=.62,
 axis_weight_4=.40,axis_weight_5=.34,axis_weight_6=.26,
 outflow_onset=2.70,outflow_width=.42,k_outflow=.085,
 collapse_onset=3.15,collapse_width=.35,k_overlap=3.50,overlap_decay=.025,
 k_closure_4=1.32,k_closure_5=1.46,k_closure_6=1.62,closure_relaxation=.030,k_collapse_drain=.86,
 fraction_particle=.34,fraction_rank_memory=.16,fraction_expansion=.24,
 fraction_prestructure_return=.16,fraction_dissipation=.10,
 expansion_decay=.23,expansion_decay_to_particle=.55,lambda_h=.82,expansion_volume_floor=.75,
 c_info_particle=1.,c_info_rank=.82,comoving_particle_scale=1.55,comoving_rank_scale=2.75,
 isolation_exponent=4.,k_isolation_particle=1.10,k_isolation_rank=1.30,
 k_reconnection_particle=.025,k_reconnection_rank=.012,
 particle_connected_decay=.004,particle_isolated_decay=.001,
 rankmem_connected_decay=.003,rankmem_isolated_decay=.0008,relaxation_return_fraction=.35,
 rank_tolerance=.055,extinction_energy_threshold=.060,stability_window=1.,stability_relative_tolerance=.012)

def sg(x):
 return 1/(1+math.exp(-x)) if abs(x)<700 else float(x>0)
def win(t,on,off,w): return sg((t-on)/w)*(1-sg((t-off)/w))
def late(t,on,w): return sg((t-on)/w)
def take(src,rate,dt): return min(max(rate,0)*dt,max(src,0))
def iso(x,m):
 y=max(x,0)**m; return y/(1+y)
def gram(e,eta,rho):
 g=np.zeros((3,3)); z={a:max(e[a]*(1-eta[a]),0) for a in A}
 for i,a in enumerate(A):
  g[i,i]=z[a]
  for j in range(i+1,3):
   b=A[j]; g[i,j]=g[j,i]=math.sqrt(z[a]*z[b])*rho[(a,b)]
 return g
def rank(g,tol):
 v=np.linalg.eigvalsh(g); return int(np.sum(v>tol+1e-12)),v

def params(row):
 x=D.copy(); x.update({k:v for k,v in row.items() if not pd.isna(v)}); return NS(**x)

def sim(p):
 if abs(sum(getattr(p,k) for k in ('fraction_particle','fraction_rank_memory','fraction_expansion','fraction_prestructure_return','fraction_dissipation'))-1)>1e-12: raise ValueError('fractions')
 ep=p.initial_prestructure_energy; hc={a:0. for a in A}; hi={a:0. for a in A}
 pc=pi=rc=ri=ex=ds=0.; rho={q:0. for q in P}; eta={a:0. for a in A}
 w={4:p.axis_weight_4,5:p.axis_weight_5,6:p.axis_weight_6}; kc={4:p.k_closure_4,5:p.k_closure_5,6:p.k_closure_6}
 if abs(sum(w.values())-1)>1e-12: raise ValueError('weights')
 a=1.; ci=co=cr=cip=cir=0.; ld0=cd0=3; xp0=xr0=0.; formed=ext=False; rows=[]; ev=[]
 def event(t,c,n,fr,to,cd,H):
  ev.append(dict(scenario=p.name,time_tau=float(t),event_class=c,event=n,from_dimension=fr,to_dimension=to,connected_dimension=cd,scale_factor=a,H=H,cumulative_inflow=ci,cumulative_outflow=co,cumulative_collapse_release=cr))
 for t in np.arange(0,p.t_end+p.dt/2,p.dt):
  f=win(float(t),p.formation_onset,p.formation_offset,p.formation_width); c=late(float(t),p.collapse_onset,p.collapse_width); o=late(float(t),p.outflow_onset,p.outflow_width)
  cap=max(1-(sum(hc.values())+sum(hi.values()))/p.high_rank_capacity,0)
  q=take(ep,p.k_inflow*f*ep*cap,p.dt); ep-=q; ci+=q
  for x in A: hc[x]+=q*w[x]
  qo=0.
  for x in A:
   z=take(hc[x],p.k_outflow*o*hc[x],p.dt); hc[x]-=z; ep+=z; qo+=z
  co+=qo; ef=min((sum(hc.values())+sum(hi.values()))/p.high_rank_capacity,1)
  for x in P: rho[x]=min(max(rho[x]+p.dt*(p.k_overlap*c*ef*(1-rho[x])-p.overlap_decay*(1-c)*rho[x]),0),1)
  for x in A: eta[x]=min(max(eta[x]+p.dt*(kc[x]*c*(1-eta[x])-p.closure_relaxation*f*eta[x]),0),1)
  qr=0.
  for x in A:
   tot=hc[x]+hi[x]
   if tot<=0: continue
   z=take(tot,p.k_collapse_drain*c*eta[x]*tot,p.dt); sh=hc[x]/tot
   hc[x]-=z*sh; hi[x]-=z*(1-sh); qr+=z
  if qr:
   pc+=qr*p.fraction_particle; rc+=qr*p.fraction_rank_memory; ex+=qr*p.fraction_expansion
   ep+=qr*p.fraction_prestructure_return; ds+=qr*p.fraction_dissipation; cr+=qr
  z=take(ex,p.expansion_decay*ex,p.dt); ex-=z; pc+=z*p.expansion_decay_to_particle; ds+=z*(1-p.expansion_decay_to_particle)
  H=p.lambda_h*math.sqrt(max(ex,0)/(a**3+p.expansion_volume_floor)); vp=H*a*p.comoving_particle_scale; vr=H*a*p.comoving_rank_scale
  xp=vp/max(p.c_info_particle,1e-12); xr=vr/max(p.c_info_rank,1e-12); ip=iso(xp,p.isolation_exponent); irg=iso(xr,p.isolation_exponent)
  z=take(pc,p.k_isolation_particle*ip*pc,p.dt); pc-=z; pi+=z; cip+=z
  z=take(pi,p.k_reconnection_particle*(1-ip)*pi,p.dt); pi-=z; pc+=z
  qri=0.
  for x in A:
   z=take(hc[x],p.k_isolation_rank*irg*hc[x],p.dt); hc[x]-=z; hi[x]+=z; qri+=z
   z=take(hi[x],p.k_reconnection_rank*(1-irg)*hi[x],p.dt); hi[x]-=z; hc[x]+=z
  z=take(rc,p.k_isolation_rank*irg*rc,p.dt); rc-=z; ri+=z; qri+=z
  z=take(ri,p.k_reconnection_rank*(1-irg)*ri,p.dt); ri-=z; rc+=z; cir+=qri
  rp=rd=0.
  for name,rate in [('pc',p.particle_connected_decay),('pi',p.particle_isolated_decay),('rc',p.rankmem_connected_decay),('ri',p.rankmem_isolated_decay)]:
   val=locals()[name]; z=take(val,rate*val,p.dt)
   if name=='pc': pc-=z
   elif name=='pi': pi-=z
   elif name=='rc': rc-=z
   else: ri-=z
   rp+=z*p.relaxation_return_fraction; rd+=z*(1-p.relaxation_return_fraction)
  ep+=rp; ds+=rd; a*=math.exp(H*p.dt)
  le={x:hc[x]+hi[x] for x in A}; lr,lv=rank(gram(le,eta,rho),p.rank_tolerance); cd,cv=rank(gram(hc,eta,rho),p.rank_tolerance); ld=3+lr; cd=3+cd
  if xp0<1<=xp: event(t,'isolation_onset','particle_expansion_speed_exceeds_c_info',ld,ld,cd,H)
  elif xp0>=1>xp: event(t,'isolation_release_window','particle_expansion_speed_below_c_info',ld,ld,cd,H)
  if xr0<1<=xr: event(t,'isolation_onset','rank_expansion_speed_exceeds_c_info',ld,ld,cd,H)
  elif xr0>=1>xr: event(t,'isolation_release_window','rank_expansion_speed_below_c_info',ld,ld,cd,H)
  xp0,xr0=xp,xr
  if ld!=ld0: event(t,'formation' if ld>ld0 else 'collapse',f'D{ld0}_to_D{ld}_local_rank',ld0,ld,cd,H); ld0=ld
  if cd!=cd0: event(t,'connected_formation' if cd>cd0 else 'connectivity_loss',f'D{cd0}_to_D{cd}_connected_rank',cd0,cd,cd,H); cd0=cd
  ph=pc+pi; rm=rc+ri; hh=sum(hc.values())+sum(hi.values()); re=hh+ph+rm+ex; total=ep+re+ds
  if re>=4*p.extinction_energy_threshold: formed=True
  if formed and not ext and t>p.formation_offset and re<p.extinction_energy_threshold: event(t,'extinction','formed_episode_energy_below_extinction_threshold',ld,ld,cd,H); ext=True
  pf=pi/ph if ph else 0.; rt=hh+rm; rf=(sum(hi.values())+ri)/rt if rt else 0.
  phase='extinct' if re<p.extinction_energy_threshold and t>p.formation_offset else ('formation' if f>.2 and q>qo+qr else ('collapse' if qr/p.dt>.02 else ('expansion_isolation' if xp>1 or xr>1 else 'stabilization')))
  rows.append(dict(scenario=p.name,t_tau=float(t),phase_label=phase,prestructure_energy=ep,high_rank_connected_energy=sum(hc.values()),high_rank_isolated_energy=sum(hi.values()),rank_memory_connected_energy=rc,rank_memory_isolated_energy=ri,particle_connected_energy=pc,particle_isolated_energy=pi,expansion_energy=ex,dissipated_energy=ds,inflow_rate=q/p.dt,outflow_rate=qo/p.dt,net_high_rank_flux_rate=(q-qo)/p.dt,collapse_release_rate=qr/p.dt,cumulative_inflow=ci,cumulative_outflow=co,cumulative_net_high_rank_flux=ci-co,cumulative_collapse_release=cr,scale_factor=a,volume_factor=a**3,H=H,e_fold=math.log(a),particle_speed_ratio=xp,rank_speed_ratio=xr,particle_isolated_fraction=pf,rank_assigned_isolated_fraction=rf,local_dimension=ld,connected_dimension=cd,realized_episode_energy=re,energy_ledger_error=total-p.initial_prestructure_energy))
 df=pd.DataFrame(rows); tt=df.t_tau.to_numpy(); df['a_dot']=np.gradient(df.scale_factor.to_numpy(),tt); df['a_ddot']=np.gradient(df.a_dot.to_numpy(),tt); df['accelerated_expansion']=df.a_ddot>0
 n=max(round(p.stability_window/p.dt),2); tail=df.tail(n); re=float(df.iloc[-1].realized_episode_energy); var=(tail.realized_episode_energy.max()-tail.realized_episode_energy.min())/max(tail.realized_episode_energy.mean(),1e-12)
 st='extinct_episode' if re<p.extinction_energy_threshold else ('stable' if var<=p.stability_relative_tolerance else 'relaxing')
 s=dict(scenario=p.name,terminal_state=st,final_local_dimension=int(df.iloc[-1].local_dimension),final_connected_dimension=int(df.iloc[-1].connected_dimension),event_count=len(ev),event_sequence=' -> '.join(x['event'] for x in ev) or 'none',total_prestructure_to_highrank_inflow=ci,total_highrank_to_prestructure_outflow=co,net_highrank_energy_import=ci-co,total_collapse_release=cr,final_scale_factor=float(df.iloc[-1].scale_factor),total_e_fold=float(df.iloc[-1].e_fold),peak_H=float(df.H.max()),accelerated_expansion_duration_tau=float(df.accelerated_expansion.sum()*p.dt),final_particle_isolated_fraction=float(df.iloc[-1].particle_isolated_fraction),peak_particle_isolated_fraction=float(df.particle_isolated_fraction.max()),final_rank_assigned_isolated_fraction=float(df.iloc[-1].rank_assigned_isolated_fraction),peak_rank_assigned_isolated_fraction=float(df.rank_assigned_isolated_fraction.max()),final_realized_episode_energy=re,final_prestructure_energy=float(df.iloc[-1].prestructure_energy),final_dissipated_energy=float(df.iloc[-1].dissipated_energy),max_energy_ledger_error=float(df.energy_ledger_error.abs().max()))
 return df,pd.DataFrame(ev),s

def grid():
 r=[]
 for i,c,x in product((.55,.75,1.,1.25,1.5),(.08,.20,.45,.85,1.30),(.45,.70,1.,1.30,1.65)):
  p=params(dict(name=f'grid_i{i:.2f}_c{c:.2f}_x{x:.2f}',dt=.004,k_inflow=.62*i,k_overlap=3.5*c,k_closure_4=1.32*c,k_closure_5=1.46*c,k_closure_6=1.62*c,k_collapse_drain=.86*c,lambda_h=.82*x)); _,_,s=sim(p); r.append(dict(inflow_scale=i,collapse_scale=c,expansion_scale=x,**s))
 return pd.DataFrame(r)
def conv():
 r=[]
 for d in (.008,.004,.002,.001):
  _,_,s=sim(params(dict(name=f'convergence_{d}',dt=d))); r.append(dict(dt=d,total_e_fold=s['total_e_fold'],peak_H=s['peak_H'],final_particle_isolated_fraction=s['final_particle_isolated_fraction'],final_rank_assigned_isolated_fraction=s['final_rank_assigned_isolated_fraction'],final_local_dimension=s['final_local_dimension'],final_connected_dimension=s['final_connected_dimension'],max_energy_ledger_error=s['max_energy_ledger_error']))
 return pd.DataFrame(r)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input-config',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--write-grid',action='store_true'); z=ap.parse_args(); z.output.mkdir(parents=True,exist_ok=True)
 ps=[params(r) for r in pd.read_csv(z.input_config).to_dict('records')]; ss=[]; ee=[]
 for p in ps:
  d,e,s=sim(p); d.to_csv(z.output/f'd6_fcs_timeseries_{p.name}.csv',index=False); ss.append(s)
  if len(e): ee.append(e)
 pd.DataFrame(ss).to_csv(z.output/'d6_fcs_scenario_summary.csv',index=False); pd.concat(ee,ignore_index=True).to_csv(z.output/'d6_fcs_event_log.csv',index=False); conv().to_csv(z.output/'d6_fcs_convergence.csv',index=False)
 if z.write_grid:
  g=grid(); g.to_csv(z.output/'d6_fcs_parameter_grid_125.csv',index=False); g.groupby(['terminal_state','final_local_dimension','final_connected_dimension']).size().reset_index(name='cell_count').sort_values('cell_count',ascending=False).to_csv(z.output/'d6_fcs_parameter_grid_summary.csv',index=False)
 (z.output/'manifest.json').write_text(json.dumps(dict(author='Kwon Dominicus',classification='Construction / Consistency Check',prestructure='N-dimensional candidate/exchange regime with finite normalized energy',scenarios=[vars(p) for p in ps],summaries=ss),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
