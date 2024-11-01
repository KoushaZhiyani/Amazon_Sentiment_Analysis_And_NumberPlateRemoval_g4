def extract_argument_value(args, key_name, default=None):
    expected_prefix = f"--{key_name}="
    expected_flag = f"--{key_name}"

    for i, arg in enumerate(args):
        if arg.startswith(expected_prefix):
            return arg[len(expected_prefix):]

        elif arg == expected_flag and i + 1 < len(args):
            return args[i + 1]

    if default is not None:
        return default
    else:
        raise ValueError(f"Argument '--{key_name}' is mandatory and not provided in arguments.")