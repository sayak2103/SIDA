<div align="center">
<img src="./images/SIDA.png" alt="Image Alt Text" width="150" height="150">
<h3> SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model </h3>

  <p align="center">
    <a href='https://arxiv.org/pdf/2412.04292'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'> </a>
    <a href='https://hzlsaber.github.io/projects/SIDA/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'> </a>
    <a href='#' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Huggingface%20Model-8A2BE2' alt='Project Page'> </a>
    <a href='#' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Dataset-Coming%20Soon-yellow' alt='Dataset Coming Soon'>
    <a href='https://www.youtube.com/watch?v=oAc9BxOoDe8&t=2s' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Video-Watch%20Now-red' alt='Video'> </a>
  </p>
</div>


[Zhenglin Huang](https://scholar.google.com/citations?user=30SRxRAAAAAJ&hl=en&oi=ao), [Jinwei Hu](https://orcid.org/0009-0008-5261-211X), [Xiangtai Li](https://lxtgh.github.io/), [Yiwei He](https://orcid.org/0000-0003-0717-8517), [Xingyu Zhao](https://www.xzhao.me/supervision-teaching)
[Bei Peng](https://beipeng.github.io/), [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home), [Xiaowei Huang](https://cgi.csc.liv.ac.uk/~xiaowei/), [Guangliang Cheng](https://sites.google.com/view/guangliangcheng/homepage)


## Abstract
The rapid advancement of generative models in creating highly realistic images poses substantial risks for misinformation dissemination. For instance, a synthetic image, when shared on social media, can mislead extensive audiences and erode trust in digital content, resulting in severe repercussions.
Despite some progress, academia has not yet created a large and diversified deepfake detection dataset for social media, nor has it devised an effective solution to address this issue. In this paper, we introduce the **Social media Image Detection dataSet (SID-Set)**, which offers three key advantages:
1. **Extensive volume**: Featuring 300K AI-generated/tampered and authentic images with comprehensive annotations.
2. **Broad diversity**: Encompassing fully synthetic and tampered images across various classes.
3. **Elevated realism**: Including images that are predominantly indistinguishable from genuine ones through mere visual inspection.

Furthermore, leveraging the exceptional capabilities of large multimodal models, we propose a new image deepfake detection, localization, and explanation framework, named **SIDA (Social media Image Detection, localization, and explanation Assistant)**. SIDA not only discerns the authenticity of images but also delineates tampered regions through mask prediction and provides textual explanations of the model’s judgment criteria.
Compared with state-of-the-art deepfake detection models on SID-Set and other benchmarks, extensive experiments demonstrate that SIDA achieves superior performance.

## News
- 🔥 The code and dataset are coming soon
- 🔥 (26-02-2025) SIDA has been accepted by CVPR2025! The code and dataset will be released recently.
## Methods

<div align="center">
  <figcaption><strong>Figure 1: Generation Process</strong></figcaption>
  <img src="images/generation.png" width="100%">
</div>

<div align="center">
  <figcaption><strong>Figure 2: Model Pipeline Overview</strong></figcaption>
  <img src="images/Pipeline.png" width="100%">
</div>

## Experiment

<p align="center"> <img src="images/experiment.png" width="100%"> </p>

## Dataset Access

We provide two methods to access the SID_Set dataset:

1. Public Access via Hugging Face
The full training set and validation set are publicly available on [Hugging Face](https://huggingface.co/datasets/saberzl/SID_Set). To load SID_Set, follow the instructions in the [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets/index). You can also explore visual examples directly through the dataset viewer on the platform.

2. Download from Google Drive
Alternatively, you can download the train.zip and val.zip files for SID_Set from [Google Drive](https://drive.google.com/drive/folders/1sFZxSrDibjpvzTrHeNVS1fTf-qIue744?usp=drive_link). Due to size limitations, we’ve split the train_full_synthetic set into two parts. After downloading, please place both parts in a train/ directory.

For the test set, we provide only a single test.zip file to minimize the risk of data contamination (e.g., from foundation models crawling the test set for training). You can download it [here](https://drive.google.com/file/d/1M2GGCvpg7UDpP6HaHT18-VqzIyD5yhC3/view?usp=drive_link).

## Citation 

```
@misc{huang2025sidasocialmediaimage,
      title={SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model}, 
      author={Zhenglin Huang and Jinwei Hu and Xiangtai Li and Yiwei He and Xingyu Zhao and Bei Peng and Baoyuan Wu and Xiaowei Huang and Guangliang Cheng},
      year={2025},
      booktitle={Conference on Computer Vision and Pattern Recognition}
}
```

## Acknowledgement
-  This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) and [LISA](https://github.com/dvlab-research/LISA). 