import os
from pyscf import gto, lo
from pyscf.dft import rks, gen_grid
import numpy as np

mol = gto.Mole()
mol.atom = '''
C 0.0 0.0 0.0
H 0.0 0.0 0.5
H 0.0 0.0 -0.5
'''
mol.unit = 'Ang'
mol.charge = 0
mol.verbose = 5

mol.basis = {}
mol.ecp = {}

with open('H.aug-cc-pVTZ.nwchem') as fb:
    bas_str = fb.read()
mol.basis['H'] = gto.basis.parse(bas_str)
with open('H.ccECP.nwchem') as fe:
    ecp_str = fe.read()
mol.ecp['H'] = gto.basis.parse_ecp(ecp_str)
with open('C.aug-cc-pVTZ.nwchem') as fb:
    bas_str = fb.read()
mol.basis['C'] = gto.basis.parse(bas_str)
with open('C.ccECP.nwchem') as fe:
    ecp_str = fe.read()
mol.ecp['C'] = gto.basis.parse_ecp(ecp_str)
mol.build()

title = 'ch2'

def run_postTreatment(mf, energy, title=title):
    dm_cpu = mf.make_rdm1()
    mf.grids.level = 5
    mf.grids.build()
    grad = mf.nuc_grad_method().kernel()
    dip = mf.dip_moment(unit='DEBYE', dm=dm_cpu)
    quad = mf.quad_moment(unit='DEBYE-ANG', dm=dm_cpu)
    s = mf.mol.intor_symmetric('int1e_ovlp')
    lowdin_ao = lo.orth_ao(mf.mol, 'lowdin')
    dm_lowdin = lowdin_ao.T @ dm_cpu @ lowdin_ao
    pop_lowdin = np.einsum('ii->i', dm_lowdin)
    print("pop_lowdin=", pop_lowdin)
    mayer_bond_indices = get_pyscf_MBO(mf.mol, mf)
    print("mayer_bond=", mayer_bond_indices)
    print("dip=", dip)
    print("quad=", quad)
    print(f"Final SCF converged energy: {energy:.10f}")


def get_pyscf_MBO(mol, mf):
    import traceback, logging
    try:
        dm_cpu = mf.make_rdm1()
        dm = dm_cpu
        s = mol.intor_symmetric('int1e_ovlp')
        natom = mol.natm
        PS = dm @ s

        shellbf = mol.aoslice_by_atom()
        ao_breaks = []
        for x in range(natom):
            ao_end = shellbf[x][3]
            ao_breaks.append(ao_end)

        mayer_bond_indices = np.zeros((natom, natom), dtype='float32')

        def atom_index_for_ao(i):
            for a, end_ao in enumerate(ao_breaks):
                if i < end_ao:
                    return a
            return natom - 1

        for i in range(len(PS)):
            a_i = atom_index_for_ao(i)
            for j in range(i):
                a_j = atom_index_for_ao(j)
                if a_i == a_j:
                    continue
                val = PS[i, j] * PS[j, i]
                mayer_bond_indices[a_i, a_j] += val
                mayer_bond_indices[a_j, a_i] += val

    except Exception as e:
        logging.error(traceback.format_exc())
        return None

    return mayer_bond_indices


mf = rks.RKS(mol).density_fit()
mf.with_df.auxbasis = {'default': 'aug-cc-pvtz-jkfit', 'I':'def2-universal-jkfit'}
mf.xc = 'wb97m-d3bj'
mf.chkfile = title + '.chk'
mf.conv_tol = 1e-10
mf.direct_scf_tol = 1e-14
mf.diis_space = 12
mf.diis_start_cycle = 5
mf.init_guess = ''

mf.level_shift = 0.1
mf.damp = 0.0
mf.grids.level = 3
mf.max_cycle = 300

energy = mf.kernel()

if mf.converged:
    run_postTreatment(mf, energy, title=title)
else:
    raise Exception('SCF not converged!')

