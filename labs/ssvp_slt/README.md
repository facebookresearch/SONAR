<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>  [![report](https://img.shields.io/badge/ArXiv-Paper-blue)](https://arxiv.org/abs/2402.09611)


## SSVP-SLT: Self-supervised Video Pretraining for Sign Language Translation

This repository contains research code for the paper [*Towards Privacy-Aware Sign Language Translation at Scale*](https://arxiv.org/abs/2402.09611).

<p align="middle">
  <img src=".github/ssvp_slt_overview.png"  alt="SSVP-SLT Overview">
</p>

<p align="middle">
  <img width=50% src=".github/ssvp_slt_language_supervised.png"  alt="SSVP-SLT Overview">
</p>



SSVP-SLT relies on masked autoencoding (MAE) on anonymized videos as a form of self-supervised pretraining to learn continuous sign language representations at scale. The learned representations are transferred to the supervised gloss-free sign language translation task. SSVP-SLT outperforms prior SOTA methods on the ASL-to-English How2Sign benchmark in the finetuned and zero-shot settings by over 3 BLEU points. 

----

### Installation

We provide installation instructions in [INSTALL.md](INSTALL.md).

### Usage
#### 1. Preparing the data

We describe how to prepare the datasets in [DATASETS.md](DATASETS.md).

#### 2. Pretraining

- MAE pretraining instructions are in [pretraining/README.md](pretraining/README.md). 
- Joint MAE & CLIP/FLIP pretraining instructions are in [pretraining_clip/README.md](pretraining_clip/README.md).

#### 3. Sign Language Translation (SLT)

Instructions for feature extraction and SLT training and evaluation are in [translation/README.md](translation/README.md).

---- 
### DailyMoth-70h

We release the DailyMoth-70h (DM-70) dataset as part of this project. DailyMoth-70h is released under a CC-BY-NC 4.0 license.

You can find an overview of the data and download and data preparation instructions in [DATASETS.md](DATASETS.md). 

Alternatively, download the files manually via these links:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Subset</th>
<th valign="bottom">Link</th>
<th valign="bottom">md5</th>


<tr><td align="left">Raw videos</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/dailymoth-70h/raw_videos.tar.gz">download</a></td>
<td align="center"><tt>875ffe4eeac3a37e50b4202c2b4996d2</tt></td>
</tr>

<tr><td align="left">Blurred clips</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/dailymoth-70h/blurred_clips.tar.gz">download</a></td>
<td align="center"><tt>a2819c7b06a8b38eb7686e4dc90a7433</tt></td>
</tr>

<tr><td align="left">Unblurred clips</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/dailymoth-70h/unblurred_clips.tar.gz">download</a></td>
<td align="center"><tt>3e69046f6cf415cec89c3544d0523325</tt></td>
</tr>

<tr><td align="left">Manifest files</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/dailymoth-70h/manifests.tar.gz">download</a></td>
<td align="center"><tt>69e500cc5cfad3133c4b589428865472</tt></td>
</tr>
</tbody></table>


> [!NOTE]
> Check out our [paper](https://arxiv.org/abs/2402.09611) for detailed information on the DailyMoth-70h dataset.


---- 
### Citing our work
If you find our work useful in your research, please consider citing:

```bibtex
@article{rust-etal-2024-slt,
    title={Towards Privacy-Aware Sign Language Translation at Scale}, 
    author={Phillip Rust and Bowen Shi and Skyler Wang and Necati Cihan Camg\"{o}z and Jean Maillard},
    year={2024},
    journal={arXiv preprint},
    url={https://arxiv.org/abs/2402.09611},
}
```

### References
This codebase is heavily influenced by the [mae](https://github.com/facebookresearch/mae) and [mae_st](https://github.com/facebookresearch/mae_st) repositories. Our models are based on code from [Hiera](https://github.com/facebookresearch/hiera), [HF Transformers](https://github.com/huggingface/transformers), [OpenCLIP](https://github.com/mlfoundations/open_clip), and [Fairseq](https://github.com/facebookresearch/fairseq).

### License
This project is primarily under the CC-BY-NC 4.0 license; see [LICENSE](LICENSE) for details. Portions of the project are available under separate license terms: [Transformers](https://github.com/huggingface/transformers) is licensed under the [Apache-2.0 license](https://github.com/huggingface/transformers/blob/main/LICENSE) and [OpenCLIP](https://github.com/mlfoundations/open_clip) is licensed under the [OpenCLIP license](https://github.com/mlfoundations/open_clip/blob/main/LICENSE).
