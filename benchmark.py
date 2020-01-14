#!/usr/bin/env python3

import subprocess as sp
import os.path as path
import glob
import csv
import re


# CDV column names.
COLUMNS = ["Num. particules", "Tot. Simulation Time", "Mover Time / Cycle", "Interp. Time / Cycle"]

# Regex for getting results.
RE_NOP    = re.compile(r"^\+\+ TOTAL NUMBER OF PARTICULES: (\d+)", re.M)
RE_SIMTOT = re.compile(r"^   Tot. Simulation Time \(s\) = ([0-9.-]+)", re.M)
RE_MOVT   = re.compile(r"^   Mover Time / Cycle   \(s\) = ([0-9.-]+)", re.M)
RE_INTERT = re.compile(r"^   Interp. Time / Cycle \(s\) = ([0-9.-]+)", re.M)


def find_result(text, regex, cast):
    "Find a result in STDOUT."

    return cast(regex.search(text).group(1))


def run_sputniPIC(here, file):
    "Run sputniPIC program on the given file."

    print("\n------------------------------------")
    print("|| RUNNING SPUTNIPIC ON %s" % path.basename(file))
    print("------------------------------------", flush=True)

    prog = path.join(here, "bin", "sputniPIC.out")
    ps = sp.Popen([prog, file], stdout=sp.PIPE)
    out, _ = ps.communicate()

    text = out.decode()
    results = [
        find_result(text, RE_NOP, int),
        find_result(text, RE_SIMTOT, float),
        find_result(text, RE_MOVT, float),
        find_result(text, RE_INTERT, float)
    ]

    print("  Num. of particuls = %s" % results[0])
    print("  Tot. Sim Time (s) = %s" % results[1])
    print("------------------------------------\n", flush=True)

    return results


def main():
    "Entry point of this program."

    here = path.abspath(path.dirname(__file__))
    files = glob.glob(path.join(here, "inputfiles", "Bench*.inp"))
    files.sort()

    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(COLUMNS)

        for file in files:
            writer.writerow(run_sputniPIC(here, file))


if __name__ == "__main__":
    main()
