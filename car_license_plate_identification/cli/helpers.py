import sys


def extract_argument_value(args, key_name):
    expected_prefix = f"--{key_name}="
    expected_flag = f"--{key_name}"

    for index, arg in enumerate(args):
        if arg.startswith(expected_prefix):
            return arg[len(expected_prefix):]

        elif arg == expected_flag and index + 1 < len(args):
            return args[index + 1]

    raise ValueError(f"Argument '--{key_name}' not found in arguments.")