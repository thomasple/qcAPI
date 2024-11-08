#!/usr/bin/env python3
import psi4

import argparse
import numpy as np
import time
from functools import partial
import multiprocessing as mp
import os

import requests


BOHR = 0.52917721067

PERIODIC_TABLE_STR = """
H                                                                                                                           He
Li  Be                                                                                                  B   C   N   O   F   Ne
Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
"""

PERIODIC_TABLE = ["Dummy"] + PERIODIC_TABLE_STR.strip().split()

PERIODIC_TABLE_REV_IDX = {s: i for i, s in enumerate(PERIODIC_TABLE)}

print_flush = partial(print, flush=True)


def compute_entry(record, num_threads=1, maxiter=150):
    conformation = record["conformation"]
    method = record["method"]
    basis = record["basis"]
    restricted = record["restricted"]
    id = record["id"]

    start_time = time.time()
    try:

        psi4.set_output_file(f"{id}.out", False)
        psi4.set_options({"scf__maxiter": maxiter})
        if restricted:
            psi4.set_options({'scf__reference': 'RHF'})
        else:
            psi4.set_options({'scf__reference': 'UHF'})
            psi4.set_options({'scf__guess_mix': 'True'})
        psi4.set_options({"scf__wcombine": False})
        psi4.set_num_threads(num_threads)
        # psi4.core.be_quiet()

        coordinates = conformation["coordinates"]
        species = conformation["species"]
        symbols = [PERIODIC_TABLE[s] for s in species]
        total_charge = round(conformation.get("total_charge", 0))
        total_atomic_number = np.sum(species)
        multiplicity = 1 if (total_charge + total_atomic_number) % 2 == 0 else 2

        geom_str = f"{total_charge} {multiplicity}\nnocom\nnoreorient\n"

        for s, (x, y, z) in zip(symbols, coordinates):
            geom_str += f"{s} {x} {y} {z}\n"

        print_flush(geom_str)

        mol = psi4.geometry(geom_str)
        mol.reset_point_group("c1")

        if method == "none" or basis == "none":
            raise ValueError("Method or basis not defined")

        de, wfn = psi4.gradient(f"{method}/{basis}", molecule=mol, return_wfn=True)
        forces = -de.np / BOHR
        e = wfn.energy()
        props = [
            "dipole",
            "wiberg_lowdin_indices",
            "mayer_indices",
            "mbis_charges",
            # "MBIS_VOLUME_RATIOS",
        ]
        psi4.oeprop(wfn, *props, title="test")

        dipole = wfn.array_variable("SCF DIPOLE").np.flatten() * BOHR
        mbis_charges = wfn.array_variable("MBIS CHARGES").np.flatten()
        mbis_dipoles = wfn.array_variable("MBIS DIPOLES").np * BOHR
        mbis_quadrupoles = wfn.array_variable("MBIS QUADRUPOLES").np * BOHR**2
        mbis_octupoles = wfn.array_variable("MBIS OCTUPOLES").np * BOHR**3
        mbis_volumes = (
            wfn.array_variable("MBIS RADIAL MOMENTS <R^3>").np.flatten() * BOHR**3
        )
        # mbis_volume_ratios = wfn.array_variable("MBIS VOLUME RATIOS").np.flatten()
        mbis_valence_widths = (
            wfn.array_variable("MBIS VALENCE WIDTHS").np.flatten() * BOHR
        )
        wiberg_lowdin_indices = wfn.array_variable("WIBERG LOWDIN INDICES").np
        mayer_indices = wfn.array_variable("MAYER INDICES").np

        elapsed_time = time.time() - start_time

        output_conf = dict(
            species=species,
            coordinates=coordinates,
            total_charge=total_charge,
            energy=e,
            forces=forces.tolist(),
            dipole=dipole.tolist(),
            mbis_charges=mbis_charges.tolist(),
            mbis_dipoles=mbis_dipoles.tolist(),
            mbis_quadrupoles=mbis_quadrupoles.tolist(),
            mbis_octupoles=mbis_octupoles.tolist(),
            mbis_volumes=mbis_volumes.tolist(),
            # mbis_volume_ratios=mbis_volume_ratios.tolist(),
            mbis_valence_widths=mbis_valence_widths.tolist(),
            wiberg_lowdin_indices=wiberg_lowdin_indices.tolist(),
            mayer_indices=mayer_indices.tolist(),
        )
        output = dict(
            id=record.get("id", None),
            conformation=output_conf,
            elapsed_time=elapsed_time,
            converged=1,
            method=method,
            basis=basis,
        )
        os.system(f"rm {id}*")


    except Exception as ex:
        print_flush(f"Error computing entry: {ex}")
        elapsed_time = time.time() - start_time
        output = dict(
            id=record.get("id", None),
            conformation=conformation,
            elapsed_time=elapsed_time,
            converged=0,
            method=method,
            basis=basis,
            error=str(ex),
        )

    psi4.core.clean()

    return output


def main():
    parser = argparse.ArgumentParser(description="Populate a qcAPI database with jobs")
    parser.add_argument(
        "address",
        type=str,
        default="127.0.0.1:8000",
        help="URL:PORT of the qcAPI server",
    )
    parser.add_argument(
        "--num_threads", "-n", type=int, default=1, help="Number of threads to use"
    )
    parser.add_argument(
        "--maxiter", "-m", type=int, default=150, help="Maximum number of SCF iterations"
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=60, help="Ping frequency in seconds"
    )

    args = parser.parse_args()
    url = args.address.split(":")[0]
    port = args.address.split(":")[1]

    mp.set_start_method("spawn")

    while True:
        response = requests.get(f"http://{url}:{port}/get_next_record/")
        if response.status_code == 210:
            print_flush("No more records. Exiting.")
            break
        elif response.status_code != 200:
            print_flush("Error getting next record. Retrying in a bit.")
            time.sleep(0.5)
            continue

        body = response.json()
        record,worker_id = body
        print("WORKER ID: ", worker_id)
        print(record)

        # entry = compute_entry(record, args.num_threads, args.maxiter)
        # compute entry in a separate process asynchronously
        pool = mp.Pool(1)
        res = pool.apply_async(compute_entry, args=(record, args.num_threads, args.maxiter))
        job_already_done = False
        while not job_already_done:
            try:
                delay = np.random.uniform(0.8, 1.2) * args.delay
                entry = res.get(timeout=delay)
                break
            except mp.TimeoutError:
                response = requests.get(f"http://{url}:{port}/get_record_status/{record['id']}?worker_id={worker_id}")
                if response.status_code != 200:
                    print_flush(
                        "Error updating record. Got status code: ", response.status_code
                    )
                    continue
                job_status = response.json()
                print_flush("JOB STATUS: ", job_status)
                job_already_done = job_status == 1
        
        if job_already_done:
            print_flush("Job already done by another worker. Killing QC calculation and getting a new job.")
            # kill the process
            pool.terminate()
            pool.join()
            entry = record
                          
        response = requests.put(f"http://{url}:{port}/qc_result/{worker_id}", json=entry)
        if response.status_code != 200:
            print_flush(
                "Error updating record. Got status code: ", response.status_code
            )

    psi4.core.clean()

if __name__ == "__main__":
    main()
