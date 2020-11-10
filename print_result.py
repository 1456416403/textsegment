from argparse import ArgumentParser
import pickle
from pathlib2 import Path
parser = ArgumentParser()
parser.add_argument('--result_dir', help='where is your result of test? ', default='checkpoints/result_0')
args = parser.parse_args()


print("show result in {}".format(args.result_dir))

result_dir = Path(args.result_dir)
with open(result_dir/"best_result", "rb") as f:
    result = pickle.load(f)

print(result)


for k in result.keys():
    print(k+":")
    print(result[k])