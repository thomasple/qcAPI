import os
from pyscf import gto, lo
import traceback
import logging
from gpu4pyscf.dft import rks, gen_grid
import numpy as np

mol = gto.Mole()
mol.atom = '''
O -0.401435 1.335586 -1.535564
O 1.007661 -0.662792 1.717838
O -2.172806 -0.106826 0.611155
O 1.251017 -2.160733 -0.770482
O 0.565485 2.464562 1.484120
O 3.199612 0.490122 0.593871
O 1.949262 2.383173 -1.501694
O -0.230955 -1.905088 -3.067891
O -1.020967 0.377457 3.684430
O -1.855406 3.532803 -0.864416
O -1.744743 2.975640 2.896569
O -3.522115 1.633872 -2.430745
O 2.569949 1.067290 3.685306
O -4.386345 1.430215 0.346183
O -3.109020 -3.178186 1.328955
O -2.183096 -0.398630 -4.142350
O -0.262209 2.086830 -4.314435
O 3.281384 0.603614 -3.566131
O 0.583590 4.819393 -0.837771
O 2.619208 -2.026568 3.680396
O -2.249884 -1.940762 4.034113
O 2.529174 -2.398926 -3.668008
O 2.649859 3.860516 2.253027
O 1.215608 -5.091228 -0.659475
O -3.692502 3.748291 1.213108
O 5.237405 -1.544256 -0.051131
O -5.043077 -2.250023 -1.220382
O -5.042688 0.639331 2.882774
O 4.543976 3.823637 0.086844
O -1.432900 -5.826182 0.389068
H -0.574772 0.621092 -0.920864
H 0.408483 1.774186 -1.262879
H 1.690971 -0.286207 1.143020
H 0.387154 -1.091380 1.100528
H -2.039752 -0.591223 1.430825
H -3.013352 0.447744 0.676389
H 0.649037 -2.107171 -1.569350
H 2.148446 -2.018223 -1.107513
H 0.508240 2.499579 0.493772
H 1.437775 2.833471 1.768821
H 3.013077 1.381475 0.318838
H 3.913043 0.245268 -0.025012
H 1.721636 3.291399 -1.663041
H 2.387164 2.034299 -2.277486
H -0.506224 -1.089937 -3.466762
H -0.880411 -2.479353 -3.433248
H -0.385199 0.209197 2.985372
H -1.494790 -0.459330 3.898034
H -1.512342 2.783251 -1.268824
H -1.052211 4.091704 -0.859325
H -1.696747 2.102950 3.273846
H -0.907219 2.992379 2.460161
H -3.082989 2.535747 -2.343818
H -3.908404 1.501051 -1.535621
H 2.029694 1.627122 4.259709
H 1.970182 0.869447 2.949992
H -4.556865 2.133726 0.991416
H -5.226659 1.123884 0.087399
H -3.201076 -2.725327 0.498883
H -3.975913 -3.533330 1.526767
H -2.439501 0.325407 -3.558002
H -2.727179 -1.184845 -4.157100
H -0.965635 2.694537 -4.563457
H -0.448832 1.764252 -3.427773
H 4.189527 0.579839 -3.285740
H 3.291936 0.878628 -4.499865
H 0.978036 5.089936 -0.043306
H 0.324254 5.588846 -1.323596
H 3.546746 -1.899314 3.640972
H 2.380471 -1.610730 2.893547
H -3.083167 -1.973020 4.569246
H -2.427951 -2.303864 3.179986
H 2.934057 -1.559911 -3.619239
H 1.644169 -2.300283 -3.301422
H 2.891508 3.896963 3.211192
H 2.226958 4.715654 1.980741
H 1.748097 -5.619425 -0.028611
H 1.357756 -4.120244 -0.560107
H -3.048660 3.537413 1.912335
H -3.128261 3.794505 0.433114
H 5.406046 -1.594183 0.915124
H 5.392825 -2.439957 -0.381487
H -4.930412 -1.326463 -0.900299
H -5.999037 -2.336027 -1.356966
H -4.860534 0.819179 3.803491
H -4.479051 -0.075987 2.631500
H 5.416714 3.894696 0.403571
H 3.988919 3.806429 0.858606
H -0.733658 -5.215705 0.034931
H -2.280801 -5.463149 0.208058
'''
mol.unit = 'Ang'
mol.charge = 0
mol.verbose = 5

mol.basis = {}
mol.ecp = {}
# ccecp scenario: load basis and ECP from local files in /leonardo_scratch/fast/EUHPC_D14_044/gb-atz/PseudoBasisPyscfTz/
with open('/leonardo_scratch/fast/EUHPC_D14_044/gb-atz/PseudoBasisPyscfTz/H.aug-cc-pVTZ.nwchem') as fb:
    bas_str = fb.read()
mol.basis['H'] = gto.basis.parse(bas_str)
with open('/leonardo_scratch/fast/EUHPC_D14_044/gb-atz/PseudoBasisPyscfTz/H.ccECP.nwchem') as fe:
    ecp_str = fe.read()
mol.ecp['H'] = gto.basis.parse_ecp(ecp_str)
with open('/leonardo_scratch/fast/EUHPC_D14_044/gb-atz/PseudoBasisPyscfTz/O.aug-cc-pVTZ.nwchem') as fb:
    bas_str = fb.read()
mol.basis['O'] = gto.basis.parse(bas_str)
with open('/leonardo_scratch/fast/EUHPC_D14_044/gb-atz/PseudoBasisPyscfTz/O.ccECP.nwchem') as fe:
    ecp_str = fe.read()
mol.ecp['O'] = gto.basis.parse_ecp(ecp_str)
mol.build()

title = 'water_conf_0'

def run_tiered_scf(mf, title=title, max_tiers=5):

    #mf.init_guess = 'atom'

    tier_params = [
        {"level_shift": 5.0, "damp": 0.5, "diis_space": 6, "atom_grid": (50, 302), "max_cycle": 100},
        {"level_shift": 2.5, "damp": 0.3, "diis_space": 8, "atom_grid": (99, 590), "max_cycle": 150},
        {"level_shift": 1.0, "damp": 0.2, "diis_space": 10, "atom_grid": (120, 974), "max_cycle": 200},
        {"level_shift": 0.5, "damp": 0.1, "diis_space": 12, "atom_grid": (120, 974), "max_cycle": 250},
        {"level_shift": 0.0, "damp": 0.0, "diis_space": 12, "atom_grid": (120, 974), "max_cycle": 300},
    ]

    dm = mf.make_rdm1()
    for tier, params in enumerate(tier_params, start=1):
        print(f"Tier {tier}/{max_tiers}: Applying parameters {params}")
        mf.level_shift = params["level_shift"]
        mf.damp = params["damp"]
        mf.diis_space = params["diis_space"]
        mf.grids = gen_grid.Grids(mf.mol)
        mf.grids.atom_grid = params["atom_grid"]
        mf.grids.level = 3
        mf.max_cycle = params["max_cycle"]
        mf.grids.build()
        energy = mf.kernel(dm)
        if mf.converged:
            print(f"SCF converged successfully at Tier {tier} with energy: {energy:.10f}")
            run_postTreatment(mf, energy, title=title)
            return energy
        dm = mf.make_rdm1()
    print("SCF failed to converge across all tiers.")
    return None


def run_postTreatment(mf, energy, title=title):
    dm_cpu = mf.to_cpu().make_rdm1()
    dm_gpu = mf.make_rdm1()
    dm = dm_gpu.get()
    mf.grids.level = 5
    mf.grids.build()
    grad = mf.nuc_grad_method().kernel()
    dip = mf.dip_moment(unit='DEBYE', dm=dm_gpu)
    quad = mf.quad_moment(unit='DEBYE-ANG', dm=dm_gpu)
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
    print("success - go have a cake!")



def get_pyscf_MBO(mol, mf):
    import traceback, logging
    try:
        dm_cpu = mf.to_cpu().make_rdm1()
        dm_gpu = mf.make_rdm1()
        dm = dm_gpu.get()
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


if True:  # main execution
    mf = rks.RKS(mol).density_fit()
    mf.with_df._cderi_to_save = title+'_df_ints.h5'
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
        run_tiered_scf(mf, title=title)

