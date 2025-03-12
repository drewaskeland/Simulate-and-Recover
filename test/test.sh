#!/bin/bash
# test.sh: Run the unit tests in test_file.py.

echo "Running tests..."

# Add the parent directory (where src is located) to PYTHONPATH.
export PYTHONPATH=$(pwd)/..

# Run the test file.
python3 -m unittest test_file.py || { echo "Some tests failed."; exit 1; }

echo "All tests passed!"
