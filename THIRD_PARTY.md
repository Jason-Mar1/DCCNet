# Third-party notices

This file records third-party source-code and runtime dependencies used by DCCNet. It does not grant rights beyond the upstream licenses and terms.

## Depth Anything

The `depth_anything/` source included with this release originates from [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything), which is licensed under the [Apache License 2.0](https://github.com/LiheYoung/Depth-Anything/blob/main/LICENSE). Release configurations identify `LiheYoung/depth_anything_vits14` at pinned Hugging Face revision [`1bee667edbfec7f9b345abaffec8c9c2bd5293c2`](https://huggingface.co/LiheYoung/depth_anything_vits14/tree/1bee667edbfec7f9b345abaffec8c9c2bd5293c2). The pretrained asset is not bundled; the runtime may download its pinned `pytorch_model.bin` on an actual run unless the user supplies a local depth checkpoint.

## DINOv2

DINOv2 source is **not vendored** in this release. At runtime, the dependency loader obtains the official [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) source at pinned commit [`7764ea0f912e53c92e82eb78a2a1631e92725fc8`](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8). The upstream source is licensed under the [Apache License 2.0](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE).

## Weights and data

No pretrained DCCNet checkpoint, third-party pretrained weight, or benchmark dataset is redistributed in this repository. Pretrained weights, model cards, and datasets may carry their own licenses, terms of use, access controls, attribution requirements, or redistribution restrictions. Users are responsible for reviewing and complying with the applicable upstream terms before downloading, using, or redistributing them.

## Project license

The original DCCNet source in this repository is distributed under the [Apache License 2.0](LICENSE). The presence of that license does not replace the notices or terms above for third-party material.
