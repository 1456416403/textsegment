from argparse import ArgumentParser
import pickle
from pathlib2 import Path
parser = ArgumentParser()
parser.add_argument('--result_dir', help='where is your result of test? ', default='test_result')
args = parser.parse_args()


print("show result in {}".format(args.result_dir))

result_dir = Path("test_result")
with open(result_dir , "rb") as f:
    result = pickle.load(f)

print(result)

cot = 0
for k in result.keys():
    cot += 1
    print(k+":")
    print(result[k])
print("一共有{}个测试样例".format(cot))