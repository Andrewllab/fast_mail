import csv

import numpy as np

IN_FILE_PATH = "/home/nic/Downloads/wandb_export_2026-03-23T12_08_00.034+01_00.csv"
ORDER = [
    "OpenSingleDoor",
    "OpenDoubleDoor",
    "CloseSingleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "CoffeePressButton",
    "CoffeeServeMug",
    "CoffeeSetupMug",
    "PnPCabToCounter",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "PnPCounterToSink",
    "PnPCounterToStove",
    "PnPMicrowaveToCounter",
    "PnPSinkToCounter",
    "PnPStoveToCounter",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "TurnOnStove",
    "TurnOffStove",
]


def Pakt_Pat_mode():
    results = {}
    with open(IN_FILE_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        name_idx = header.index("data.env.env_name")
        pakt_idx = header.index(
            "data.gpu_batch_transforms.t02_to_pakt_action_obs.include_tracked_in_action"
        )
        success_idx = header.index("success.max")
        print(header)
        for row in reader:
            env_name = row[name_idx]
            pakt = bool(row[pakt_idx] == "true")
            success = float(row[success_idx]) * 100
            if env_name not in results:
                results[env_name] = {"pakt": [], "pat": []}
            if pakt:
                results[env_name]["pakt"].append(success)
            else:
                results[env_name]["pat"].append(success)

    print(" ========= PAKT ===========")
    for env_name in ORDER:
        env_results = results[env_name]
        pakt_mean = np.mean(env_results["pakt"])
        pakt_std_dev = np.std(env_results["pakt"])

        print(f"{env_name}, {pakt_mean:.1f} ({pakt_std_dev:.1f})")

    print(" ========= PAT ===========")
    for env_name in ORDER:
        env_results = results[env_name]
        pat_mean = np.mean(env_results["pat"])
        pat_std_dev = np.std(env_results["pat"])

        print(f"{env_name}, {pat_mean:.1f} ({pat_std_dev:.1f})")

    pass


def Kat_mode():
    results = {}
    with open(IN_FILE_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        name_idx = header.index("data.env.env_name")
        success_idx = header.index("success.max")
        print(header)
        for row in reader:
            env_name = row[name_idx]
            success = float(row[success_idx]) * 100
            if env_name not in results:
                results[env_name] = []

            results[env_name].append(success)

    print(" ========= KAT ===========")
    for env_name in ORDER:
        if env_name not in results:
            print(f"{env_name}, {-1:.1f} ({-1:.1f})")
            continue
        env_results = results[env_name]
        pat_mean = np.mean(env_results)
        pat_std_dev = np.std(env_results)

        print(f"{env_name}, {pat_mean:.1f} ({pat_std_dev:.1f})")

    pass


if __name__ == "__main__":
    # Pakt_Pat_mode()
    Kat_mode()
