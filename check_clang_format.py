#!/usr/bin/env python3

import difflib
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def check(path, check_id):
    with open(path) as f:
        original = f.read()
    formatted = subprocess.check_output(["clang-format", "-assume-filename", path],
                                        input=original, text=True)
    diff = difflib.unified_diff(original.splitlines(keepends=True),
                                formatted.splitlines(keepends=True),
                                path,
                                "formatted")
    diff_result = list(diff)

    def result(nr_checks):
        logging.info(f"[{check_id}/{nr_checks}] checked clang-format: {path}")
        if not diff_result:
            return True
        logging.warning("differences found:")
        sys.stdout.writelines(diff_result)
        print()
        return False

    return result


if __name__ == "__main__":
    logging.basicConfig(datefmt="%H:%M:%S",
                        format="[%(asctime)s][%(levelname)s] %(message)s",
                        level=logging.DEBUG,
                        stream=sys.stderr)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    git_files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        results = []
        count = 0
        for path in git_files:
            if path.endswith(".cpp") or path.endswith(".h"):
                results.append(executor.submit(check, path, count + 1))
                count += 1

        nr_checks = count
        count = 0
        exit_code = 0
        for future in results:
            if not future.result()(nr_checks):
                exit_code = 1
        exit(exit_code)
