"""Compare independent directional refinements at fixed full candidate grids."""
from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'benchmarks/independent')]
from nanoporeos.core import VT,sha256,write_json,PNP,Scenario,source_identity
from nodal_pnp import Problem
from noneq_reference import load
from driven_refinement_ordered import load as load_ordered
from nodal_pnp_ordered import NodalPNP as OrderedReference

FIELDS=['potential_V','cplus_mol_m3','cminus_mol_m3']

def sample(arrays,x,r):
    xx,rr=np.meshgrid(x,r,indexing='ij');points=np.c_[xx.ravel(),rr.ravel()]
    return [RegularGridInterpolator((arrays['x_nm'],arrays['r_nm']),arrays[f])(points).reshape(xx.shape) for f in FIELDS]

def metrics(a,b,x,r,weight):
    delta=abs(a[0]-b[0])/VT;index=np.unravel_index(np.argmax(delta),delta.shape)
    return dict(max_phi_difference_VT=float(delta.max()),
        potential_relative_volume_L2=float(np.sqrt(np.sum(weight*(a[0]-b[0])**2)/np.sum(weight*b[0]**2))),
        plus_relative_volume_L2=float(np.sqrt(np.sum(weight*(a[1]-b[1])**2)/np.sum(weight*b[1]**2))),
        minus_relative_volume_L2=float(np.sqrt(np.sum(weight*(a[2]-b[2])**2)/np.sum(weight*b[2]**2))),
        max_plus_relative_difference=float(np.max(abs(a[1]/b[1]-1))),
        max_minus_relative_difference=float(np.max(abs(a[2]/b[2]-1))),
        max_phi_x_nm=float(x[index[0]]),max_phi_r_nm=float(r[index[1]]),
        max_phi_distance_from_end_wall_rim_nm=float(np.hypot(min(x[index[0]],20-x[index[0]]),10-r[index[1]])))

def main():
    pointer=json.loads((ROOT/'results/driven_refinement_ordered/latest.json').read_text());path=ROOT/pointer['summary']
    assert sha256(path)==pointer['sha256'];saved=json.loads(path.read_text());assert saved['status']=='FINISHED'
    directory=path.parent;protocol=json.loads((directory/'provenance/driven_refinement_ordered_protocol.json').read_text())
    references={};hashes={}
    base=ROOT/'results/noneq_reference/reference_v+100_256x128'
    for grid,parent in [((256,128),base),*[(tuple(r['grid']),ROOT/r['path']) for r in saved['records']]]:
        problem=Problem(nx=grid[0],nr=grid[1],**protocol['problem'])
        metadata,arrays=(load if grid==(256,128) else load_ordered)(parent,problem)
        assert metadata['status']=='PASS';references[grid]=(metadata,arrays);hashes[str(parent.relative_to(ROOT))]=sha256(parent/'manifest.json')
        model=OrderedReference(problem)
        phi=arrays['potential_V'].ravel()/VT
        mup=np.log(arrays['cplus_mol_m3'].ravel()/100)+phi;mum=np.log(arrays['cminus_mol_m3'].ravel()/100)-phi
        y=np.r_[phi[model.interior_phi],mup[model.interior_ions],mum[model.interior_ions]]
        audit=model.audit(y);assert audit['coupled_scaled_residual_max']<1e-10 and audit['positive']
        assert audit['section_current_spread_A']<=1e-18+1e-6*abs(audit['current_A'])
        for field in ['coupled_scaled_residual_max','current_A','section_current_spread_A']:
            assert np.isclose(audit[field],metadata['audit'][field],rtol=1e-7,
                atol=2e-13 if field=='coupled_scaled_residual_max' else 1e-18)
    rows=[];candidate_rows=[]
    for nx,nr in protocol['fixed_candidate_evaluation_grids']:
        x=(np.arange(nx)+.5)*20/nx;r=(np.arange(nr)+.5)*10/nr
        weight=np.broadcast_to(np.pi*((r+5/nr)**2-(r-5/nr)**2)*20/nx,(nx,nr))
        values={g:sample(a,x,r) for g,(_,a) in references.items()}
        for coarse,fine,label in [((256,128),(512,128),'axial'),((256,128),(256,256),'radial'),
                ((256,128),(512,256),'both'),((512,128),(512,256),'radial_after_axial'),((256,256),(512,256),'axial_after_radial')]:
            ma=references[coarse][0];mb=references[fine][0]
            rows.append(dict(evaluation_grid=f'{nx}x{nr}',path=label,coarse=str(coarse),fine=str(fine),
                **metrics(values[coarse],values[fine],x,r,weight),
                current_relative_difference=abs(ma['audit']['current_A']-mb['audit']['current_A'])/max(abs(mb['audit']['current_A']),1e-18)))
        candidate=ROOT/f'results/noneq_reference/candidate_v+100_{nx}x{nr}'
        metadata=json.loads((candidate/'result.json').read_text())
        assert metadata['status']=='PASS' and metadata['source_sha256']==source_identity()
        for name in ['fields.npz','result.json']:hashes[str((candidate/name).relative_to(ROOT))]=sha256(candidate/name)
        with np.load(candidate/'fields.npz',allow_pickle=False) as arrays:
            assert np.allclose(arrays['x_nm'],x) and np.allclose(arrays['r_nm'],r)
            fields=[arrays[f] for f in FIELDS]
        model=PNP(Scenario(nx=nx,nr=nr,**protocol['problem']))
        y=np.array([fields[0]/VT,np.log(fields[1]/100),np.log(fields[2]/100)]).ravel()
        audit=model.audit(y);assert audit['coupled_residual_max']<=2e-11 and audit['positive'] and audit['current_consistent']
        candidate_rows.append(dict(evaluation_grid=f'{nx}x{nr}',**metrics(fields,values[(512,256)],x,r,weight)))
    summary=dict(status='PASS_SAVED_DIRECTIONAL_COMPARISON',reference_summary_sha256=sha256(path),
        protocol_sha256=saved['protocol_sha256'],input_manifest_hashes=hashes,reference_refinements=rows,
        candidate_vs_refined_reference=candidate_rows,reference_error_bound=False,original_gates_changed=False,
        full_domain_qualified=False,scope='One +100mV/R10nm/100mM/-15mV case; empirical refinement sensitivity, not physical validation')
    output=ROOT/'results/driven_refinement_v2_analysis';output.mkdir(exist_ok=True)
    write_json(output/'summary.json',summary)
    for name,table in [('reference_refinements.csv',rows),('candidate_comparisons.csv',candidate_rows)]:
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    for name,table in [('Supplemental_Table_S10_Driven_Reference_Refinements.csv',rows),('Supplemental_Table_S11_Driven_Candidate_Comparisons.csv',candidate_rows)]:
        with (ROOT/'manuscript/tables'/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.5),layout='constrained');positions=np.arange(3)
    for i,(label,color) in enumerate([('axial','#71899b'),('radial','#c47756'),('both','#26968b')]):
        values=[r['max_phi_difference_VT'] for r in rows if r['path']==label]
        axes[0].bar(positions+(i-1)*.23,values,width=.23,color=color,label=label.capitalize())
    axes[0].set_yscale('log');axes[0].set_title('A  Directional reference sensitivity',fontsize=10);axes[0].legend(fontsize=8)
    axes[1].plot(positions,[r['max_phi_difference_VT'] for r in candidate_rows],'o-',color='#26968b',label='Candidate /512×256 reference')
    axes[1].axhline(.02,color='#c47756',ls='--',label='Original 0.02 VT limit')
    axes[1].set_title('B  Candidate / refined reference',fontsize=10);axes[1].legend(fontsize=7.5)
    for ax in axes:
        ax.set_xticks(positions,['36×18','72×36','144×72']);ax.set_xlabel('Fixed candidate evaluation grid');ax.set_ylabel('Maximum potential difference / VT')
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'manuscript/figures/Supplemental_Figure_S3_Driven_Refinement.{suffix}',dpi=300)
    plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
