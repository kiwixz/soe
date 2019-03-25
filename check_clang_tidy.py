#!/usr/bin/env python3

import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def check(path, check_id):
    errors = None
    try:
        subprocess.check_output(["clang-tidy", "-quiet", "-p", build_dir, "-warnings-as-errors", "*", path],
                                stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as ex:
        errors = ex.output

    def result(nr_checks):
        logging.info(f"[{check_id}/{nr_checks}] checked clang-tidy: {path}")
        if not errors:
            return True
        logging.warning("errors found:")
        print(errors, '\n')
        return False

    return result


if __name__ == "__main__":
    logging.basicConfig(datefmt="%H:%M:%S",
                        format="[%(asctime)s][%(levelname)s] %(message)s",
                        level=logging.DEBUG,
                        stream=sys.stderr)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    build_dir = sys.argv[1] if len(sys.argv) > 1 else "build"
    if not (Path(build_dir) / "compile_commands.json").exists():
        logging.critical("compile_commands.json not found, did you pass the correct build directory ?")
        exit(2)

    exit_code = 0
    git_files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        results = []
        count = 0
        for path in git_files:
            if path.endswith(".cpp"):
                results.append(executor.submit(check, path, count + 1))
                count += 1

        nr_checks = count
        count = 0
        exit_code = 0
        for future in results:
            if not future.result()(nr_checks):
                exit_code = 1
        exit(exit_code)
