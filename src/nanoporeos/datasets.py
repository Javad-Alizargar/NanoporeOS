"""Allowlisted, integrity-checked local NPZ/JSON/CSV exports, never pickle."""
from dataclasses import asdict
from pathlib import Path
import csv
import io
import json
import os
import zipfile
import numpy as np
from .core import PNP, Scenario, VT, environment, source_identity, sha256, write_json


def save_result(directory, model, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    phi,up,um = result['y'].reshape(3,model.s.nx,model.s.nr)
    tmp = directory/'fields.tmp.npz'
    np.savez_compressed(tmp, x_nm=model.x, r_nm=model.r, potential_V=phi*VT,
        cplus_mol_m3=model.s.salt_mM*np.exp(up),cminus_mol_m3=model.s.salt_mM*np.exp(um))
    os.replace(tmp,directory/'fields.npz')
    summary={k:v for k,v in result.items() if k!='y'}
    summary.update(schema_version='1.0',model_equations='steady monovalent PNP; no flow',
        environment=environment(),source_sha256=source_identity(),
        units=dict(x_nm='nm',r_nm='nm',potential_V='V',cplus_mol_m3='mol/m^3',cminus_mol_m3='mol/m^3'),
        field_layout='[axial_cell, radial_cell]',experimental_validation='NOT_PERFORMED',
        unsupported=['fluid coupling','molecule feedback','physical time dependence','instrument calibration'],
        license='UNRESOLVED: user approval required before redistribution')
    write_json(directory/'result.json',summary)
    with (directory/'sections.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['axial_face_index','current_A'])
        writer.writerows(enumerate(result['diagnostics']['section_currents_A']))
    write_json(directory/'manifest.json',dict(schema_version='1.0', files={
        name:sha256(directory/name) for name in ['fields.npz','result.json','sections.csv']}))
    return summary


def load_result(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    expected={'fields.npz','result.json','sections.csv'}
    if set(manifest['files'])!=expected:
        raise ValueError('Unexpected export manifest members')
    for name,digest in manifest['files'].items():
        if sha256(directory/name)!=digest:
            raise ValueError('Export integrity failure: '+name)
    summary=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if summary.get('schema_version')!='1.0' or summary.get('status') not in ['PASS','FAIL']:
        raise ValueError('Unsupported result schema/status')
    s=Scenario(**summary['scenario'])
    with zipfile.ZipFile(directory/'fields.npz') as z:
        if sum(m.file_size for m in z.infolist())>16*1024*1024:
            raise ValueError('Field storage limit')
    with np.load(directory/'fields.npz',allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    if set(arrays)!={'x_nm','r_nm','potential_V','cplus_mol_m3','cminus_mol_m3'}:
        raise ValueError('Unexpected fields')
    for name in ['potential_V','cplus_mol_m3','cminus_mol_m3']:
        if arrays[name].shape!=(s.nx,s.nr) or not np.all(np.isfinite(arrays[name])):
            raise ValueError('Invalid field shape or values')
    for name,n,extent in [('x_nm',s.nx,s.length_nm),('r_nm',s.nr,s.radius_nm)]:
        values=arrays[name]
        if values.shape!=(n,) or not np.all(np.isfinite(values)) or not np.allclose(values,(np.arange(n)+.5)*extent/n,rtol=1e-12,atol=1e-12):
            raise ValueError('Invalid cell-centre coordinate schema')
    if min(arrays['cplus_mol_m3'].min(),arrays['cminus_mol_m3'].min())<=0:
        raise ValueError('Nonpositive ion concentration')
    return summary,arrays


def export_zip(directory):
    load_result(directory)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['fields.npz','result.json','sections.csv','manifest.json']:
            archive.write(Path(directory)/name,name)
    return buf.getvalue()
