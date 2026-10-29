'''Figure 1: the nominal free return in the rotating frame, and the definition
of the entry flight-path angle.

Run from the repository root (after retarget_nominal.py):  python scripts/make_premise_figure.py
'''

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Arc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent)) # scripts/, for plot_style

from freereturn.constants import MU as mu
from freereturn.trajectory import tli_state
from freereturn.entry import fly_to_entry
from plot_style import plt, save

nominal = json.loads((ROOT/'results'/'nominal.json').read_text())
res = fly_to_entry(tli_state(nominal['delta_v_kms'], nominal['phi_rad']), nominal['dt_nd'], store=True)
P = res['u'][::10, :2]

fig,(a,b)=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={"width_ratios":[1.7,1]})
a.plot(P[:,0],P[:,1],color="#1f4e79",lw=1.2,label="nominal free return")
a.add_patch(Circle((-mu,0),0.03,color="#2e75b6")); a.add_patch(Circle((1-mu,0),0.012,color="#7f7f7f"))
a.plot(0,0,"k+",ms=6)
a.annotate("Earth",(-mu,0),(-0.12,-0.12),fontsize=9,arrowprops=dict(arrowstyle="-",lw=0.5))
a.annotate("Moon",(1-mu,0),(0.9,-0.16),fontsize=9,arrowprops=dict(arrowstyle="-",lw=0.5))
a.annotate("barycenter",(0,0),(0.12,0.12),fontsize=8,arrowprops=dict(arrowstyle="-",lw=0.5))
a.annotate("TLI from 185 km\nparking orbit",(-mu,-0.02),(0.18,-0.32),fontsize=8,arrowprops=dict(arrowstyle="->",lw=0.6))
k=np.argmin(np.hypot(P[:,0]-1+mu,P[:,1]))
a.annotate("lunar flyby",P[k],(0.68,0.22),fontsize=8,arrowprops=dict(arrowstyle="->",lw=0.6))
a.annotate("entry interface",(-mu,0.02),(-0.32,0.25),fontsize=8,arrowprops=dict(arrowstyle="->",lw=0.6))
a.add_patch(FancyArrowPatch((0.78,-0.45),(0.98,-0.45),arrowstyle="->",mutation_scale=10,lw=0.8))
a.add_patch(FancyArrowPatch((0.78,-0.45),(0.78,-0.25),arrowstyle="->",mutation_scale=10,lw=0.8))
a.text(0.99,-0.47,"$x$",fontsize=9); a.text(0.76,-0.23,"$y$",fontsize=9)
a.text(0.72,-0.53,r"frame rotates at unit rate",fontsize=7.5)
a.set_aspect("equal"); a.set_xlim(-0.45,1.15); a.set_ylim(-0.55,0.4)
a.set_xlabel("$x$ (nondimensional)"); a.set_ylabel("$y$ (nondimensional)")
a.set_title("(a) Planar CR3BP, rotating Earth–Moon frame",fontsize=10); a.legend(loc="upper left",fontsize=8)
for s in ["top","right"]: a.spines[s].set_visible(False)
# (b) entry flight path angle definition: close-up at the entry point
R=3.0; H=0.45
t=np.linspace(np.pi/2-0.55,np.pi/2+0.55,200)
xs=R*np.cos(t); ys=-R-H+R*np.sin(t); b.fill(np.r_[xs,xs[-1],xs[0]],np.r_[ys,-1.55,-1.55],color="#2e75b6",alpha=0.85,lw=0)
b.plot((R+H)*np.cos(t),-R-H+(R+H)*np.sin(t),"k--",lw=0.8)
b.text(-1.25,-1.05,"Earth",fontsize=9,color="w")
b.text(-1.65,0.12,r"entry interface, $|\mathbf{r}_E|=r_{\mathrm{EI}}$ (400 kft)",fontsize=8)
b.add_patch(FancyArrowPatch((0,-1.5),(0,0),arrowstyle="-|>",mutation_scale=10,lw=0.9))
b.text(0.06,-0.95,r"$\mathbf{r}_E$ (from Earth's center)",fontsize=9)
b.plot([0,1.7],[0,0],"k:",lw=0.9); b.text(1.25,0.07,"local horizontal",fontsize=8)
g=np.radians(-22); v=1.6*np.array([np.cos(g),np.sin(g)])
b.add_patch(FancyArrowPatch((0,0),v,arrowstyle="-|>",mutation_scale=12,lw=1.4,color="#c00000"))
b.text(v[0]+0.05,v[1]-0.08,r"$\mathbf{v}_E$",fontsize=11,color="#c00000")
b.add_patch(Arc((0,0),1.2,1.2,theta1=np.degrees(g),theta2=0,lw=0.9))
b.text(0.95,-0.2,r"$\gamma_{\mathrm{EI}}<0$",fontsize=9.5)
b.plot(0,0,"ko",ms=3)
b.text(-1.65,-1.85,r"safe corridor: $-7.4^\circ \leq \gamma_{\mathrm{EI}} \leq -5.25^\circ$",fontsize=8.5)
b.set_aspect("equal"); b.set_xlim(-1.7,2.3); b.set_ylim(-1.95,0.6); b.axis("off")
b.set_title("(b) Entry flight-path angle (angle exaggerated)",fontsize=10)
for ax in (a, b):
    ax.grid(False)
save(fig, 'premise_geometry')
