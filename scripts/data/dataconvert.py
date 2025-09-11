import json

import argparse
parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
parser.add_argument('--datap', default="./deepscaler/data/orz_math_57k_collected.json",
                   help='Local directory to save processed datasets')
parser.add_argument('--savepath', default="./deepscaler/data/orzmath/orz_math_57k_collected.json")
args = parser.parse_args()
datap = args.datap
# data = json.load(open(datap))
data = []
with open(datap, 'r') as f:
    for line in f:
        data.append(json.loads(line))

newdata = [{'problem': d['problem'], 'answer': d['answer']} for d in data]
#save to a new json file
savepath=args.savepath
json.dump(newdata, open(savepath, 'w'))