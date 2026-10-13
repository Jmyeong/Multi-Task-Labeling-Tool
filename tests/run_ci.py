"""Run tests and attach readable failure tracebacks to GitHub check annotations."""

import sys
import unittest


def main():
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    for test, traceback in result.failures + result.errors:
        title = test.id()
        module = title.split(".")[0]
        message = traceback.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error file=tests/{module}.py,line=1,title={title}::{message}", flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
