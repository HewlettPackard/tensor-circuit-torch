#!/usr/bin/env python3
import argparse
from typing import Type, Any

def get_CLI_input(argument: str, type: Type = str, default: Any = None):

    """Get the command-line input argument of type, or insert the default"""

    parser = argparse.ArgumentParser()
    parser.add_argument(argument, type=type, default=default)
    args, _ = parser.parse_known_args()

    # Extract attribute name from '--config_file' → 'config_file'
    arg_name = argument.lstrip('-').replace('-', '_')

    return getattr(args, arg_name)