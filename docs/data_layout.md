# Local data layout

DCCNet does not redistribute ShanghaiTech, UCF-QNRF, or JHU-CROWD++. Download each dataset from its official provider, keep the raw archive outside the repository if preferred, and create the processed directories consumed by the release YAML files.

```text
data/
|-- shanghaitech/
|   |-- part_a/                 # ShanghaiTech Part A (A)
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   `-- part_b/                 # ShanghaiTech Part B (B)
|       |-- train/
|       |-- val/
|       `-- test/
|-- ucf_qnrf/                   # UCF-QNRF (Q)
|   |-- train/
|   |-- val/
|   `-- test/
`-- jhu_crowdpp/                # JHU-CROWD++
    |-- train/
    |-- val/
    `-- test/

splits/
|-- shanghaitech/
|   |-- part_a_train.txt        # tracked
|   |-- part_a_val.txt          # tracked
|   |-- part_b_train.txt        # tracked
|   `-- part_b_val.txt          # tracked
|-- ucf_qnrf/
|   |-- train.txt               # tracked
|   `-- val.txt                 # tracked
`-- jhu/
    |-- str_train.txt            # user-provided
    |-- str_val.txt              # user-provided
    |-- str_test.txt             # user-provided
    |-- sta_train.txt            # user-provided
    |-- sta_val.txt              # user-provided
    `-- sta_test.txt             # user-provided
```

For each processed image, the dataset readers expect a coordinate array with the same stem. Training additionally requires a density map:

```text
data/shanghaitech/part_a/train/IMG_1.jpg
data/shanghaitech/part_a/train/IMG_1.npy          # point coordinates, shape [N, 2]
data/shanghaitech/part_a/train/IMG_1_dmap.npy     # training density map
```

The test readers use the image and matching `*.npy` point array. `*_dmap.npy` is required by the training dataset.

## Convert official data

The tracked ShanghaiTech and UCF-QNRF splits are consumed by `--split-dir`; do not substitute an untracked random split when comparing with the paper protocol.

```bash
# ShanghaiTech Part A and Part B
python utils/preprocess_data.py --dataset sta --origin-dir /path/to/part_A --data-dir data/shanghaitech/part_a --split-dir splits
python utils/preprocess_data.py --dataset stb --origin-dir /path/to/part_B --data-dir data/shanghaitech/part_b --split-dir splits

# UCF-QNRF
python utils/preprocess_data.py --dataset qnrf --origin-dir /path/to/UCF-QNRF --data-dir data/ucf_qnrf --split-dir splits

# JHU-CROWD++
python utils/preprocess_data.py --dataset jhu --origin-dir /path/to/jhu_crowd_v2.0 --data-dir data/jhu_crowdpp
```

Generate the required density maps after conversion. Run this only for a processed root whose image/point pairs are available locally:

```bash
python utils/dmap_gen.py --path data/shanghaitech/part_a
python utils/dmap_gen.py --path data/shanghaitech/part_b
python utils/dmap_gen.py --path data/ucf_qnrf
python utils/dmap_gen.py --path data/jhu_crowdpp
```

## JHU-CROWD++ manifests

The release includes `splits/jhu/.gitkeep` rather than a domain assignment. Create the six required files from official JHU metadata and the exact Street/Stadium protocol you choose. Each line must be an image path **relative** to the corresponding processed phase directory; for example, `0001.jpg` denotes `data/jhu_crowdpp/train/0001.jpg` in `str_train.txt`.

The paper's JHU results are Street -> Stadium and Stadium -> Street. Because the required domain manifests are not distributed here, the JHU commands are not turn-key until you provide them. Do not infer or redistribute annotations that the dataset provider has not authorized.

Official sources: [ShanghaiTech](https://github.com/desenzhou/ShanghaiTechDataset), [UCF-QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/), and [JHU-CROWD++](https://www.crowd-counting.com/).
