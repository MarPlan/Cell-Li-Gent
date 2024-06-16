import argparse
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--in_put", type=str)
args = parser.parse_args()
print(args.in_put)
