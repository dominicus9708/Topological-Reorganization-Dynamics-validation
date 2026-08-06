from __future__ import annotations

import argparse, json, math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

Q=(4,5,6); P={4:np.array([1.,0.,0.]),5:np.array([0.,1.,0.]),6:np.array([0.,0.,1.])}

@dataclass(frozen=True)
class Par:
 name:str; dt:float=.002; t_end:float=18.; initial_prestructure_energy:float=12.; cross_channel_capacity:float=4.8; initial_particle_energy:float=1.
 inflow_onset:float=.8; inflow_offset:float=6.; inflow_width:float=.35; k_inflow:float=.52; channel_weight_4:float=1/3; channel_weight_5:float=1/3; channel_weight_6:float=1/3
 k_cross_to_d3:float=.72; cross_energy_return_rate:float=.012; cross_energy_return_fraction:float=.9
 outflow_onset:float=4.2; outflow_offset:float=13.; outflow_width:float=.5; k_outflow_cross:float=.1; k_outflow_relax:float=.055
 relaxation_decay_connected:float=.055; relaxation_decay_isolated:float=.018; relaxation_return_fraction:float=.2
 convergence_memory_gain:float=.82; convergence_memory_decay:float=.34
 baseline_axial_tension:float=1.; k_tension_relax_inflow:float=.24; k_tension_converge_outflow:float=.31; tension_equalization:float=.52; tension_restore:float=.075
 lambda_relaxation:float=.92; lambda_convergence:float=.88; lambda_anisotropy:float=.34; lambda_tension_relaxation:float=.42; lambda_tension_convergence:float=.46; hubble_relaxation:float=.42; expansion_volume_floor:float=.85
 c_info_particle:float=1.; c_info_channel:float=.86; comoving_particle_scale:float=1.55; comoving_channel_scale_4:float=2.1; comoving_channel_scale_5:float=2.45; comoving_channel_scale_6:float=2.8
 isolation_exponent:float=4.; k_isolation_particle:float=1.1; k_isolation_channel:float=1.35; k_reconnection_particle:float=.03; k_reconnection_channel:float=.018
 particle_domain_overlap_4:float=0.; particle_domain_overlap_5:float=0.; particle_domain_overlap_6:float=0.
 active_channel_energy_threshold:float=.03; anisotropy_event_threshold:float=.045; expansion_event_threshold:float=.02; contraction_event_threshold:float=-.02; extinction_energy_threshold:float=.06; stability_window:float=1.5; stability_relative_tolerance:float=.015

def sg(x): return 1/(1+math.exp(-x)) if abs(x)<700 else float(x>0)
def win(t,on,off,w): return sg((t-on)/w)*(1-sg((t-off)/w))
def take(src,rate,dt): return min(max(rate,0)*dt,max(src,0))
def iso(r,m):
 x=max(r,0)**m; return x/(1+x)
def rowpar(r):
 d=asdict(Par(str(r.get('name','scenario')))); d.update({k:v for k,v in r.items() if k in d and not pd.isna(v)}); d['name']=str(d['name']); return Par(**d)
def weights(p):
 d={4:p.channel_weight_4,5:p.channel_weight_5,6:p.channel_weight_6}; z=sum(d.values());
 if z<=0 or min(d.values())<0: raise ValueError('channel weights')
 return {q:d[q]/z for q in Q}
def scale(p,q): return {4:p.comoving_channel_scale_4,5:p.comoving_channel_scale_5,6:p.comoving_channel_scale_6}[q]
def overlap(p,q): return {4:p.particle_domain_overlap_4,5:p.particle_domain_overlap_5,6:p.particle_domain_overlap_6}[q]

def sim(p,keep=True):
 w=weights(p); pre=p.initial_prestructure_energy; cc={q:0. for q in Q}; ci={q:0. for q in Q}; rc=ri=ds=dpi=0.; pc=p.initial_particle_energy; pi=0.; tau=np.full(3,p.baseline_axial_tension); cm=np.zeros(3); a=1.; H=0.
 cin=cout=ct=dpt=ipr=0.; ev=[]; rows=[]; prp=0.; prq={q:0. for q in Q}; pa=pn=0; an=ex=co=False; activated=False
 def event(t,cl,n,ac,nc): ev.append(dict(scenario=p.name,time_tau=float(t),event_class=cl,event=n,base_dimension=3,active_cross_channels=ac,connected_cross_channels=nc,scale_factor=a,H=H,cumulative_inflow=cin,cumulative_outflow=cout))
 for t in np.arange(0,p.t_end+p.dt/2,p.dt):
  fi=win(float(t),p.inflow_onset,p.inflow_offset,p.inflow_width); fo=win(float(t),p.outflow_onset,p.outflow_offset,p.outflow_width); cap=max(1-(sum(cc.values())+sum(ci.values()))/p.cross_channel_capacity,0)
  qi={q:0. for q in Q}
  for q in Q:
   rq=max(H,0)*a*scale(p,q)/max(p.c_info_channel,1e-12); z=take(pre,p.k_inflow*fi*pre*cap*w[q]*(1-iso(rq,p.isolation_exponent)),p.dt); pre-=z; cc[q]+=z; qi[q]=z; cin+=z
  tr={q:0. for q in Q}; dr={q:0. for q in Q}
  for q in Q:
   z=take(cc[q],p.k_cross_to_d3*cc[q],p.dt); o=min(max(overlap(p,q),0),1); d=z*o; u=z-d; cc[q]-=z; rc+=u; pc+=d; dpi+=d; tr[q]=u; dr[q]=d; ct+=u; dpt+=d
  qo={q:0. for q in Q}
  for q in Q:
   x=take(cc[q],p.k_outflow_cross*fo*cc[q],p.dt); cc[q]-=x; y=take(rc,p.k_outflow_relax*fo*rc*w[q],p.dt); rc-=y; z=x+y; pre+=z; qo[q]=z; cout+=z
  for q in Q:
   z=take(cc[q],p.cross_energy_return_rate*cc[q],p.dt); cc[q]-=z; pre+=z*p.cross_energy_return_fraction; ds+=z*(1-p.cross_energy_return_fraction)
   z=take(ci[q],.35*p.cross_energy_return_rate*ci[q],p.dt); ci[q]-=z; pre+=z*p.cross_energy_return_fraction; ds+=z*(1-p.cross_energy_return_fraction)
  iv=sum((tr[q]/p.dt)*P[q] for q in Q); ov=sum((qo[q]/p.dt)*P[q] for q in Q); mt=float(tau.mean()); dtau=-p.k_tension_relax_inflow*iv+p.k_tension_converge_outflow*ov+p.tension_equalization*(mt-tau)+p.tension_restore*(p.baseline_axial_tension-tau); tau+=p.dt*dtau; cm+=p.dt*(p.convergence_memory_gain*ov-p.convergence_memory_decay*cm)
  for name,rate in [('rc',p.relaxation_decay_connected),('ri',p.relaxation_decay_isolated)]:
   val=rc if name=='rc' else ri; z=take(val,rate*val,p.dt)
   if name=='rc': rc-=z
   else: ri-=z
   pre+=z*p.relaxation_return_fraction; ds+=z*(1-p.relaxation_return_fraction)
  mt=float(tau.mean()); dev=float(tau.std()); rd=rc/(a**3+p.expansion_volume_floor)+p.lambda_tension_relaxation*max(p.baseline_axial_tension-mt,0); cd=float(cm.mean())+p.lambda_tension_convergence*max(mt-p.baseline_axial_tension,0); dH=p.lambda_relaxation*rd-p.lambda_convergence*cd-p.lambda_anisotropy*dev-p.hubble_relaxation*H; H+=p.dt*dH; a=max(a*math.exp(H*p.dt),1e-8)
  rp=max(H,0)*a*p.comoving_particle_scale/max(p.c_info_particle,1e-12); gp=iso(rp,p.isolation_exponent); z=take(pc,p.k_isolation_particle*gp*pc,p.dt); pc-=z; pi+=z; z=take(pi,p.k_reconnection_particle*(1-gp)*pi,p.dt); pi-=z; pc+=z
  rq={}; gq={}
  for q in Q:
   rq[q]=max(H,0)*a*scale(p,q)/max(p.c_info_channel,1e-12); gq[q]=iso(rq[q],p.isolation_exponent); z=take(cc[q],p.k_isolation_channel*gq[q]*cc[q],p.dt); cc[q]-=z; ci[q]+=z; z=take(ci[q],p.k_reconnection_channel*(1-gq[q])*ci[q],p.dt); ci[q]-=z; cc[q]+=z
  g=sum(gq.values())/3; z=take(rc,p.k_isolation_channel*g*rc,p.dt); rc-=z; ri+=z; z=take(ri,p.k_reconnection_channel*(1-g)*ri,p.dt); ri-=z; rc+=z
  ir=(abs(H)+abs(dH)+dev+float(np.linalg.norm(dtau)))*(pc+pi); ipr+=ir*p.dt; ac=sum(cc[q]+ci[q]>p.active_channel_energy_threshold for q in Q); nc=sum(cc[q]>p.active_channel_energy_threshold for q in Q)
  if ac!=pa: event(t,'cross_channel_activation' if ac>pa else 'cross_channel_deactivation',f'active_cross_channels_{pa}_to_{ac}',ac,nc); pa=ac
  if nc!=pn: event(t,'cross_channel_connection' if nc>pn else 'cross_channel_isolation',f'connected_cross_channels_{pn}_to_{nc}',ac,nc); pn=nc
  if prp<1<=rp: event(t,'particle_isolation_onset','particle_separation_speed_exceeds_information_bound',ac,nc)
  if prp>=1>rp: event(t,'particle_reconnection_window','particle_separation_speed_below_information_bound',ac,nc)
  prp=rp
  for q in Q:
   if prq[q]<1<=rq[q]: event(t,'channel_isolation_onset',f'cross_channel_{q}_separation_speed_exceeds_information_bound',ac,nc)
   if prq[q]>=1>rq[q]: event(t,'channel_reconnection_window',f'cross_channel_{q}_separation_speed_below_information_bound',ac,nc)
   prq[q]=rq[q]
  if not an and dev>=p.anisotropy_event_threshold: event(t,'anisotropy_onset','d3_axial_tension_deviation_exceeds_threshold',ac,nc); an=True
  elif an and dev<.7*p.anisotropy_event_threshold: event(t,'anisotropy_relaxation','d3_axial_tension_deviation_returns_below_threshold',ac,nc); an=False
  if not ex and H>=p.expansion_event_threshold: event(t,'expansion_onset','d3_isotropic_relaxation_drives_positive_expansion',ac,nc); ex=True
  elif ex and H<.5*p.expansion_event_threshold: event(t,'expansion_relaxation','d3_expansion_rate_returns_below_threshold',ac,nc); ex=False
  if not co and H<=p.contraction_event_threshold: event(t,'convergence_onset','prestructure_outflow_drives_d3_convergence',ac,nc); co=True
  elif co and H>.5*p.contraction_event_threshold: event(t,'convergence_relaxation','d3_convergence_rate_returns_below_threshold',ac,nc); co=False
  xe=sum(cc.values())+sum(ci.values())+rc+ri; activated|=xe>=4*p.extinction_energy_threshold; pt=pc+pi; cf=pi/pt if pt else 0; xt=sum(cc.values())+sum(ci.values()); xf=sum(ci.values())/xt if xt else 0; rt=rc+ri; rf=ri/rt if rt else 0; led=pre+xt+rt+ds+dpi-p.initial_prestructure_energy
  if keep: rows.append(dict(scenario=p.name,t_tau=float(t),base_dimension=3,prestructure_energy=pre,relaxation_connected_energy=rc,relaxation_isolated_energy=ri,inflow_rate=sum(qi.values())/p.dt,outflow_rate=sum(qo.values())/p.dt,tension_x=tau[0],tension_y=tau[1],tension_z=tau[2],mean_axial_tension=mt,axial_tension_deviation=dev,scale_factor=a,H=H,e_fold=math.log(a),particle_speed_ratio=rp,particle_isolated_fraction=cf,cross_energy_isolated_fraction=xf,relaxation_energy_isolated_fraction=rf,active_cross_channels=ac,connected_cross_channels=nc,total_exchange_energy=xe,indirect_particle_response_rate=ir,energy_ledger_error=led))
 df=pd.DataFrame(rows) if keep else None
 if keep:
  final=df.iloc[-1]; tail=df.tail(max(int(round(p.stability_window/p.dt)),2)); var=(tail.total_exchange_energy.max()-tail.total_exchange_energy.min())/max(tail.total_exchange_energy.mean(),1e-12); fe=float(final.total_exchange_energy); fh=float(final.H); fa=float(final.axial_tension_deviation); fp=float(final.particle_isolated_fraction); fx=float(final.cross_energy_isolated_fraction); sf=float(final.scale_factor); pH=float(df.H.max()); mH=float(df.H.min()); pdev=float(df.axial_tension_deviation.max()); me=float(df.energy_ledger_error.abs().max())
 else:
  fe=xe; fh=H; fa=dev; fp=cf; fx=xf; sf=a; pH=max([e['H'] for e in ev]+[H]); mH=min([e['H'] for e in ev]+[H]); pdev=max([p.anisotropy_event_threshold if e['event_class']=='anisotropy_onset' else 0 for e in ev]+[dev]); me=abs(led); var=1
 state='scale_floor_contracted_d3' if sf<=1e-6 else ('exchange_extinct_stable_d3' if fe<p.extinction_energy_threshold else ('converging_d3' if fh<=p.contraction_event_threshold else ('expansion_isolated_d3' if fp>.5 or fx>.5 else ('anisotropic_d3' if fa>=p.anisotropy_event_threshold else ('stable_exchange_d3' if var<=p.stability_relative_tolerance else 'relaxing_d3')))))
 def first(cl): return next((e['time_tau'] for e in ev if e['event_class']==cl),None)
 s=dict(scenario=p.name,base_dimension=3,terminal_state=state,event_count=len(ev),event_sequence=' -> '.join(e['event'] for e in ev) or 'none',max_active_cross_channels=max([e['active_cross_channels'] for e in ev]+[ac]),final_active_cross_channels=ac,final_connected_cross_channels=nc,total_prestructure_inflow=cin,total_prestructure_outflow=cout,net_prestructure_import=cin-cout,total_cross_to_d3_indirect_transfer=ct,total_direct_particle_import=dpt,direct_to_indirect_transfer_ratio=dpt/max(ct,1e-12),cumulative_indirect_particle_response=ipr,final_scale_factor=sf,total_e_fold=math.log(sf),final_H=fh,peak_H=pH,minimum_H=mH,first_expansion_time_tau=first('expansion_onset'),first_convergence_time_tau=first('convergence_onset'),first_channel_isolation_time_tau=first('channel_isolation_onset'),first_particle_isolation_time_tau=first('particle_isolation_onset'),peak_axial_tension_deviation=pdev,final_mean_axial_tension=mt,final_axial_tension_deviation=fa,final_particle_isolated_fraction=fp,final_cross_energy_isolated_fraction=fx,final_exchange_energy=fe,final_prestructure_energy=pre,final_dissipated_energy=ds,max_energy_ledger_error=me)
 return df,pd.DataFrame(ev),s

def conv(b):
 r=[]
 for d in (.008,.004,.002,.001):
  _,_,s=sim(Par(**{**asdict(b),'name':f'convergence_{d}','dt':d})); r.append({k:s[k] for k in ('total_e_fold','peak_H','minimum_H','peak_axial_tension_deviation','final_particle_isolated_fraction','final_cross_energy_isolated_fraction','max_energy_ledger_error')}|{'dt':d})
 return pd.DataFrame(r)
def grid(b):
 r=[]
 for i,o,x in product((.15,.35,.6,1.,1.4),(.5,1.,2.,3.,4.),(.2,.4,.7,1.,1.4)):
  p=Par(**{**asdict(b),'name':f'grid_i{i:.2f}_o{o:.2f}_x{x:.2f}','dt':.01,'t_end':16.,'k_inflow':b.k_inflow*i,'k_outflow_cross':b.k_outflow_cross*o,'k_outflow_relax':b.k_outflow_relax*o,'convergence_memory_gain':b.convergence_memory_gain*o,'lambda_relaxation':b.lambda_relaxation*x,'c_info_particle':1.2,'c_info_channel':1.4,'comoving_particle_scale':1.2,'comoving_channel_scale_4':1.2,'comoving_channel_scale_5':1.4,'comoving_channel_scale_6':1.6,'k_reconnection_channel':.08})
  _,_,s=sim(p,False); r.append(dict(inflow_scale=i,outflow_scale=o,expansion_scale=x,**s))
 return pd.DataFrame(r)

def main():
 a=argparse.ArgumentParser(); a.add_argument('--input-config',type=Path,required=True); a.add_argument('--output',type=Path,required=True); a.add_argument('--write-grid',action='store_true'); z=a.parse_args(); z.output.mkdir(parents=True,exist_ok=True); ps=[rowpar(r) for r in pd.read_csv(z.input_config).to_dict('records')]; ss=[]; ee=[]
 for p in ps:
  d,e,s=sim(p); d.to_csv(z.output/f'd3_prestructure_timeseries_{p.name}.csv',index=False); ss.append(s); ee.append(e)
 pd.DataFrame(ss).to_csv(z.output/'d3_prestructure_scenario_summary.csv',index=False); pd.concat(ee,ignore_index=True).to_csv(z.output/'d3_prestructure_event_log.csv',index=False); conv(ps[0]).to_csv(z.output/'d3_prestructure_convergence.csv',index=False)
 if z.write_grid:
  g=grid(ps[0]); g.to_csv(z.output/'d3_prestructure_parameter_grid_125.csv',index=False); g.groupby(['terminal_state','final_active_cross_channels','final_connected_cross_channels']).size().reset_index(name='cell_count').sort_values('cell_count',ascending=False).to_csv(z.output/'d3_prestructure_parameter_grid_summary.csv',index=False)
 (z.output/'manifest.json').write_text(json.dumps(dict(author='Kwon Dominicus',classification='Construction / Consistency Check',base_dimension=3,particle_coupling='default direct coupling zero',summaries=ss),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
