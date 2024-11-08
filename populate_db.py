import argparse
import requests
import pickle
from fastapi.encoders import jsonable_encoder
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Populate a qcAPI database with jobs')
    parser.add_argument('filenames', type=str, nargs='+', help='Filenames of the pickled configurations')
    parser.add_argument('--address','-a', type=str, default="127.0.0.1:8000", help='URL:PORT of the qcAPI server')
    parser.add_argument('--method','-m', type=str, default="wb97m-d3bj" ,help='Method to use')
    parser.add_argument('--basis','-b', type=str, default="def2-tzvppd",help='Basis to use')

    args = parser.parse_args()
    url = args.address.split(":")[0]
    port = args.address.split(":")[1]

    conformations = []
    for filename in args.filenames:
        with open(filename, 'rb') as f:
            conformations += pickle.load(f)
    conformations = jsonable_encoder(conformations,custom_encoder={np.ndarray: lambda x: x.tolist()})
    print(len(conformations))
    print(conformations[0])

    response = requests.post(f"http://{url}:{port}/populate/{args.method}/{args.basis}", json=conformations).json()
    print(response["message"])
    id = response["ids"][0]
    record = requests.get(f"http://{url}:{port}/get_record/{id}").json()
    print(record)

if __name__ == "__main__":
    main()
