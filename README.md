# V-Umlaut real-data experiments

This is a fork of [kocurvik/threeview](https://github.com/kocurvik/threeview) used for the real-data experiments of the paper "Linear Fundamental Matrix Estimation from 7 or 5 Points" (CVPR 2026 - [paper link](https://openaccess.thecvf.com/content/CVPR2026/html/Kucukpinar_Linear_Fundamental_Matrix_Estimation_from_7_or_5_Points_CVPR_2026_paper.html), code: [CIVA-Lab/v-umlaut](https://github.com/CIVA-Lab/v-umlaut)).
It contains only the two-view evaluation `eval_twoview.py` for `5pF-V-Umlaut`, `5pF-V-Umlaut + ENM (5p)` and the reference `5pE`, and `plot_dual.py`, which draws the runtime vs. accuracy figures.

## Installation

Create an environment with numpy, h5py, tqdm, prettytable and matplotlib.

Install [PoseLib fork with V-Umlaut in branch v-umlaut](https://github.com/CIVA-Lab/PoseLib/tree/v-umlaut) into the environment:
```shell
git clone https://github.com/CIVA-Lab/PoseLib
cd PoseLib
git checkout v-umlaut
pip install .
```

## Data preparation

The raw data:
- PhotoTourism: the 13 scenes of the [Image Matching Challenge 2020 training data](https://www.cs.ubc.ca/~kmyi/imw2020/data.html), i.e. the 10 training scenes and `sacre_coeur`, `st_peters_square`, `reichstag` in training format, extracted to `/path/to/phototourism/<scene>/dense/` (`prepare_im.py` reads `dense/images` and `dense/sparse`).
- Aachen Day-Night v1.1 ([visuallocalization.net](https://www.visuallocalization.net/datasets/)): `images/database_and_query_images.zip` and `aachen_v1_1.zip` from the [dataset server](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/), extracted together to `/path/to/aachen/aachen_v1_1/` (`prepare_im.py` reads `images_upright` and `3D-models/aachen_v_1_1`).

Image pairs are made with `datasets/prepare_im.py` from these Colmap reconstructions (needs pytorch, LightGlue, kornia, OpenCV, scipy and a CUDA GPU).
For every scene it writes `R.h5`, `T.h5`, `K.h5`, the SuperPoint features and `pairs-a0.1-features_superpoint_noresize_2048-LG.{txt,h5}`: 5000 random image pairs whose co-visible points cover at least 10 % of both images, matched with SuperPoint + LightGlue.
Before running it make sure that the repo is in your python path (e.g. `export PYTHONPATH=/path/to/repo/threeview`).
```shell
python datasets/prepare_im.py -n 5000 -a 0.1 /path/to/phototourism-output /path/to/phototourism
python datasets/prepare_im.py -n 5000 -a 0.1 /path/to/aachen-output /path/to/aachen
```

## Reproducing the figures

`eval_twoview.py` reads `R.h5`, `T.h5`, `K.h5`, `<matches>.txt` (one image pair per line) and `<matches>.h5` from a prepared scene folder.
With `-g` every method runs PoseLib LO-RANSAC with a 1 px threshold and 10, 20, 50, 100, 200, 500, 1000, 2000, 5000 and 10000 iterations. Run with `-nw 1` (one process) to get meaningful runtimes. Results are saved to `results/`, figures to `figs/`.

Supplementary Figure 6 (Sacre-Coeur):
```shell
python eval_twoview.py -g -nw 1 pairs-a0.1-features_superpoint_noresize_2048-LG /path/to/phototourism-output/sacre_coeur
python plot_dual.py -t sacre_coeur results/twoview-sacre_coeur-graph-pairs-a0.1-features_superpoint_noresize_2048-LG.json
```

Figure 5 (a), all 13 PhotoTourism scenes pooled:
```shell
jsons=""
for scene in brandenburg_gate buckingham_palace colosseum_exterior grand_place_brussels notre_dame_front_facade palace_of_westminster pantheon_exterior reichstag sacre_coeur st_peters_square taj_mahal temple_nara_japan trevi_fountain; do
    python eval_twoview.py -g -nw 1 pairs-a0.1-features_superpoint_noresize_2048-LG /path/to/phototourism-output/$scene
    jsons="$jsons results/twoview-$scene-graph-pairs-a0.1-features_superpoint_noresize_2048-LG.json"
done
python plot_dual.py -t phototourism $jsons
```

Figure 5 (b), Aachen Day-Night v1.1:
```shell
python eval_twoview.py -g -nw 1 pairs-a0.1-features_superpoint_noresize_2048-LG /path/to/aachen-output/aachen_v1_1
python plot_dual.py -t aachen results/twoview-aachen_v1_1-graph-pairs-a0.1-features_superpoint_noresize_2048-LG.json
```

## Citation

If you use this code, you may cite our paper:
```
@InProceedings{V-Umlaut_2026_CVPR,
    author    = {Kucukpinar, Taci Ata and Mogollon, Juan and Fraser, Joshua and Duff, Timothy and Palaniappan, Kannappan},
    title     = {Linear Fundamental Matrix Estimation from 7 or 5 Points},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {21464-21473}
}
```

The PhotoTourism (Image Matching Challenge 2020) and Aachen Day-Night v1.1 datasets used in the experiments are described in:
```
@article{Jin2020,
    author={Yuhe Jin and Dmytro Mishkin and Anastasiia Mishchuk and Jiri Matas and Pascal Fua and Kwang Moo Yi and Eduard Trulls},
    title={{Image Matching across Wide Baselines: From Paper to Practice}},
    journal={International Journal of Computer Vision},
    year={2020}
}
@article{Zhang2020ARXIV,
    author={Zhang, Zichao and Sattler, Torsten and Scaramuzza, Davide},
    title={{Reference Pose Generation for Visual Localization via Learned Features and View Synthesis}},
    journal={arXiv},
    volume={2005.05179},
    year={2020}
}
@inproceedings{Sattler2018CVPR,
    author={Sattler, Torsten and Maddern, Will and Toft, Carl and Torii, Akihiko and Hammarstrand, Lars and Stenborg, Erik and Safari, Daniel and Okutomi, Masatoshi and Pollefeys, Marc and Sivic, Josef and Kahl, Fredrik and Pajdla, Tomas},
    title={{Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions}},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}
@inproceedings{Sattler2012BMVC,
    author={Sattler, Torsten and Weyand, Tobias and Leibe, Bastian and Kobbelt, Leif},
    title={{Image Retrieval for Image-Based Localization Revisited}},
    booktitle={British Machine Vision Conference (BMVC)},
    year={2012}
}
```

The pairs are matched with [SuperPoint](https://arxiv.org/abs/1712.07629) and [LightGlue](https://github.com/cvg/LightGlue), and poses are estimated with [PoseLib](https://github.com/PoseLib/PoseLib).

---

# Practical solutions to the relative pose of three calibrated cameras

The original repo contains code for paper "Practical solutions to the relative pose of three calibrated cameras" (CVPR 2025 - [paper link](https://openaccess.thecvf.com/content/CVPR2025/papers/Tzamos_Practical_Solutions_to_the_Relative_Pose_of_Three_Calibrated_Cameras_CVPR_2025_paper.pdf))

## Citation
```
@inproceedings{tzamos2025practical,
  title={Practical solutions to the relative pose of three calibrated cameras},
  author={Tzamos, Charalambos and Kocur, Viktor and Ding, Yaqing and Barath, Daniel and Haladov{\'a}, Zuzana Berger and Sattler, Torsten and Kukelova, Zuzana},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={21913--21923},
  year={2025}
}
```
