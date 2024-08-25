import csv
import os.path
import time

import psutil
from tqdm import tqdm


def monitor_system():
    """
    Function to monitor the state of the system at current time
    :return: a dictionary containing couples <indicator, value>
    """
    python_data = {}
    n_proc = psutil.cpu_count()

    # Adding timestamp
    python_data['_timestamp'] = time.time()

    # CPU Times
    tag = 'cpu_times'
    pp_data = psutil.cpu_times()
    if pp_data is not None:
        pp_dict = pp_data._asdict()
        for pp_key in pp_dict.keys():
            python_data[tag + '.' + pp_key] = pp_dict[pp_key]

    # CPU Stats
    tag = 'cpu_stats'
    pp_data = psutil.cpu_stats()
    if pp_data is not None:
        pp_dict = pp_data._asdict()
        for pp_key in pp_dict.keys():
            python_data[tag + '.' + pp_key] = pp_dict[pp_key]

    # CPU Load
    f_obj = getattr(psutil, "getloadavg", None)
    if callable(f_obj):
        tag = 'cpu_load'
        pp_data = psutil.getloadavg()
        if pp_data is not None and isinstance(pp_data, tuple) and len(pp_data) == 3:
            python_data[tag + ".load_1m"] = pp_data[0]
            python_data[tag + ".load_5m"] = pp_data[1]
            python_data[tag + ".load_15m"] = pp_data[2]

    # Swap Memory
    tag = 'swap'
    pp_data = psutil.swap_memory()
    if pp_data is not None:
        pp_dict = pp_data._asdict()
        for pp_key in pp_dict.keys():
            python_data[tag + '.' + pp_key] = pp_dict[pp_key]

    # Virtual Memory
    tag = 'virtual'
    pp_data = psutil.virtual_memory()
    if pp_data is not None:
        pp_dict = pp_data._asdict()
        for pp_key in pp_dict.keys():
            python_data[tag + '.' + pp_key] = pp_dict[pp_key]

    # Disk
    tag = 'disk'
    pp_data = psutil.disk_usage('/')
    if pp_data is not None:
        pp_dict = pp_data._asdict()
        for pp_key in pp_dict.keys():
            python_data[tag + '.' + pp_key] = pp_dict[pp_key]
            print(pp_data)

    # Disk IO
    try:
        tag = 'disk_io'
        pp_data = psutil.disk_io_counters()
        if pp_data is not None:
            pp_dict = pp_data._asdict()
            for pp_key in pp_dict.keys():
                python_data[tag + '.' + pp_key] = pp_dict[pp_key]
    except:
        err = 1

    # Net IO
    tag = 'net_io'
    pp_data = psutil.net_io_counters()
    if pp_data is not None:
        pp_dict = pp_data._asdict()
        for pp_key in pp_dict.keys():
            python_data[tag + '.' + pp_key] = pp_dict[pp_key]

    return python_data


def main_monitor(out_filename : str, max_n_obs : int = 10, obs_interval_sec : int = 1):
    """
    Main function for monitoring
    :param obs_interval_sec: seconds in between two observations
    :param out_filename: name of the output CSV file
    :param max_n_obs: maximum number of observations
    :return: no return
    """

    # Checking of out_filename already exists: if yes, delete
    if os.path.exists(out_filename):
        os.remove(out_filename)

    # Monitoring Loop
    print('Monitoring for %d times' % max_n_obs)
    obs_count = 0

    for obs_count in tqdm(range(max_n_obs), desc='Monitor Progress Bar'):
        start_time = time.time()
        obs = monitor_system()

        # Writing on the command line and as a new line of a CSV file
        with open(out_filename, "a", newline="") as csvfile:
            # Create a CSV writer using the field/column names
            writer = csv.DictWriter(csvfile, fieldnames=obs.keys())
            if obs_count == 0:
                # Write the header row (column names)
                writer.writeheader()
            writer.writerow(obs)

        # Sleeping to synchronize to the obs-interval
        exe_time_s = time.time() - start_time
        sleep_s = obs_interval_sec - exe_time_s

        # Sleep to catch up with cycle time
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            print('Warning: execution of the monitor took too long (%.3f sec)' % (exe_time_s - obs_interval_sec))
        obs_count += 1


if __name__ == "__main__":
    """
    Entry point for the Monitor
    """
    main_monitor('output_folder/monitored_data_anomaly.csv', 1000, 1.0)
