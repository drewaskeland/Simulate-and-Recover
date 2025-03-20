#Code produced with Chat GPT assistance

#!/bin/bash
# test.sh: Run the unit tests in test_file.py.

echo "Running tests..."

# Add the parent directory (where src is located) to PYTHONPATH.
export PYTHONPATH=$(pwd)/..

# Use unittest discovery to run test_file.py in the current directory.
python3 -m unittest discover -s . -p "test_file.py" || { echo "Some tests failed."; exit 1; }

echo "All tests passed!"
