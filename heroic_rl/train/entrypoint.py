"""
This script is used as entrypoint to run a serialized Python callable passed as
base64-encoded string argument, which represents a zlib-compressed serialized
callable. This script imports only a bare minimum of standard modules.
"""

import base64
import pickle
import warnings
import zlib

warnings.simplefilter(action="ignore", category=FutureWarning)  # noqa


if __name__ == "__main__":
    import os
    import argparse

    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser()
    parser.add_argument("encoded_thunk")
    args = parser.parse_args()
    thunk = pickle.loads(zlib.decompress(base64.b64decode(args.encoded_thunk)))
    thunk()
