import json

jsondata = {
    "word2vecfile": "/hdd2/data/syxu20/GoogleNews-vectors-negative300.bin",
    "choidataset": "/home/syxu20/program/pycharm/text-segmentation/data/choi",
    "wikidataset": "/hdd2/data/syxu20/wiki_727K",
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)
