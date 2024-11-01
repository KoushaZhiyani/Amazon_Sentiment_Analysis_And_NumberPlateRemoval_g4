import sys


def extract_argument_value(args, key_name):
    # Construct the expected argument prefix with double dashes
    expected_prefix = f"--{key_name}="

    # Iterate over the arguments to find the one that matches the expected prefix
    for arg in args:
        if arg.startswith(expected_prefix):
            # Extract the value after the prefix
            value = arg[len(expected_prefix):]
            return value

    # Raise an error if the key is not found
    raise ValueError(f"Argument '--{key_name}' not found in arguments.")
