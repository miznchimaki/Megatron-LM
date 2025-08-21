import json
import os, sys
import warnings
import webdataset as wds
from tqdm import tqdm


try:
    llava_pretrain_dir = sys.argv[1]
except IndexError as _:
    warnings.warn("please pass in an argument for LLaVA-Pretrain dataset")

if not os.path.exists(llava_pretrain_dir):
     raise FileNotFoundError(f"the passed in param for LLaVA-Pretrain dataset - {llava_pretrain_dir}, does not exist.")

# Paths to the dataset files
json_file = os.path.join(llava_pretrain_dir, 'blip_laion_cc_sbu_558k.json')
output = os.path.join(llava_pretrain_dir, 'wds')

if not os.path.exists(output):
    os.mkdir(output)

# Load data
with open(json_file, 'r') as f:
    data = json.load(f)

with wds.ShardWriter(os.path.join(output, 'pretrain-%d.tar'), maxcount=10000) as shard_writer:
    for entry in tqdm(data):
        with open(os.path.join(llava_pretrain_dir, 'images', entry['image']), "rb") as img_file:
                image_data = img_file.read()
        sample = {
            "__key__": entry['id'],
            "jpg": image_data,
            "json": json.dumps(entry['conversations']).encode("utf-8"),
        }
        shard_writer.write(sample)

print(f"Dataset successfully converted to wds")
