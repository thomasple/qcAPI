import argparse
import requests
import time
import tqdm
import sys

def human_time_duration(seconds:float):
    ''' Convert seconds (duration) to human readable string 
    
    from https://gist.github.com/borgstrom/936ca741e885a1438c374824efb038b3
    '''
    
    if seconds<1.:
        return f'{seconds*1000:.3g} ms'
    if seconds<10.:
        return f'{seconds:.3g} s'
        
    TIME_DURATION_UNITS = (
      ("week","s", 60 * 60 * 24 * 7),
      ("day","s", 60 * 60 * 24),
      ("h","", 60 * 60),
      ("min","", 60),
      ("s","", 1),
    )
    parts = []
    for unit, plur, div in TIME_DURATION_UNITS:
        amount, seconds = divmod(int(seconds), div)
        if amount > 0:
            parts.append(f"{amount} {unit}{plur if amount > 1 else ''}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Populate a qcAPI database with jobs")
    parser.add_argument(
        "address",
        type=str,
        default="127.0.0.1:8000",
        help="URL:PORT of the qcAPI server",
    )
    parser.add_argument(
        "--refresh", "-r", type=float, default=1.0, help="refresh rate in seconds"
    )
    parser.add_argument(
        "--worker_delay","-d",
        type=float,
        default=10.,
        help="delay for recent worker check in minutes",
    )
    parser.add_argument(
        "--b1", type=float, default=0.9, help="exponential moving average parameter"    
    )

    args = parser.parse_args()
    url = args.address.split(":")[0]
    port = args.address.split(":")[1]
    delay = args.worker_delay * 60
    api_point = f"http://{url}:{port}/?delay={delay}"

    response = requests.get(api_point)
    if response.status_code != 200:
        raise ValueError("Error getting initial response")

    body = response.json()
    pending = body["pending"]
    converged = body["converged"]
    failed = body["failed"]
    total = pending + converged + failed
    processed = converged + failed
    if args.refresh <=0:
        percent = 100 * processed / total
        recent_worker = body["recently_active_workers"]
        print(f'[{percent:.2f}%] {processed}/{total} (r,f,w = {pending},{failed},{recent_worker})')
        return
    
    try:
        with tqdm.tqdm(total=total, initial=processed, dynamic_ncols=True) as pbar:
            while True:
                response = requests.get(api_point)
                if response.status_code != 200:
                    time.sleep(args.refresh)
                    continue

                body = response.json()
                pending = body["pending"]
                converged = body["converged"]
                failed = body["failed"]
                added = converged + failed - processed
                pbar.total = pending + converged + failed
                recent_worker = body["recently_active_workers"]
                pbar.set_postfix(
                    {
                        "r,f,w": (pending,failed,recent_worker),
                    }
                )
                if added > 0:
                    pbar.update(added)
                    processed = converged + failed
                if pending <= 0:
                    break
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


# NOTES:
# - use websocket for real-time updates
# - in client, put psi4 in a multiprocessing process and in the main thread periodically check back with the server to see if the job was already compleated by another worker (and to be marked as active)
