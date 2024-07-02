## SCDA

#### This is a code implemention of the SCDA proposed in the manuscript "Source-Free Collaborative Domain Adaptation via Multi-Perspective Feature Enrichment for Functional MRI Analysis".

### 1. Unsupervised Pretraining
In this stage, we utilize 3,806 auxiliary rs-fMRI scans from ABIDE, REST-meta-MDD, and ADHD-200 to construct an fMRI feature encoder. The main idea is to encourage fMRI features of each subject generated from various augmentation perspectives to be consistent. Since these auxiliary fMRIs are acquired from multi-site studies that use different scanners and even from different diseases, this pretraining is expected to help produce a general feature encoder. Note that no label information of these data is used. 

In folder `s0_unsup_tr`, run: `main_unsup_tr.py` (change `ROOT_PATH` and `SAVE_PATH`)

After training, the pretrained model (i.e., `unsuptr_mavg_fold_0_epoch_5.pth`) is used for source model initialization.

### 2. Source Model Construction
In this stage, a pretrained source model is constructed, and the training data are labeled fMRI scans from the source domain. The source model takes full-length fMRI timeseries as input and consists of a data-feeding module, a spatiotemporal feature encoder, and a class predictor. 

In folder `s1_pretrain`, run: `main_pretrain.py` (change `ROOT_PATH`, `SAVE_PATH`, and model loading path in line 218)

After training, we obtain a well-trained source model, and the source data are no longer accessible. 

### 3. Target Model Construction
In this stage, we construct a target model and perform source-free domain adaptation with a pretrained source model and unlabeled target data. We initialize the target model using parameters of the pretrained source model to facilitate source-to-target knowledge transfer without accessing source data. The target model consists of multiple collaborative branches to dynamically capture target fMRI features from three views (i.e., window warping, receptive field manipulation, and window slicing), and a mutual-consistency constraint $L_M$ is used for target model optimization.

In folder `s2_SFUDA`, run: `main_SFUDA.py` (change `ROOT_PATH`, `SAVE_PATH`, and model loading path in line 380)

After training, we obtain a target model, which can be directly used for model inference.

### Note that model checkpoints from each stage are provided in the source code.
