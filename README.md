# CSE-847-Final



In this project, we review and explored the latest cutting-edge models that aim to accomplish these goals. While there are several research studies available, not all of them have
accessible datasets. One of the observation we had is that all
these works are done on one or two specific datasets with
small number of samples, and can easily cause the problem
of overfitting. We made effort to obtain different datasets and
performed ablation studies on the models. Specifically, we
plan to cross-validate their results using new datasets that
has not been tested by these models to assess their model’s
resilience.



You can see their result as below:



| Method/dataset | CAVE                 |                      | ICVL  |       | BGH-HS |        | ARAD-HS                |                        | KAUST-HS           |                        |
|:--------------:|:--------------------:|:--------------------:|:-----:|:-----:|:------:|:------:|:----------------------:|:----------------------:|:------------------:|:----------------------:|
|                | RMSE                 | MRAE                 | RMSE  | MRAE  | RMSE   | MRAE   | RMSE                   | MRAE                   | RMSE               | MRAE                   |
| HSCNN          | 17.01                | 2.12                 | 9.21  | 2.51  | 17.01  | 1.0190 | $\mathbf{0 . 0 2 3 5}$ | $\mathbf{0 . 7 2 4}$   | 9.01               | 2.01                   |
| SRU-Net        | 15.88                | 7.1                  | 21.28 | 4.06  | 15.88  | 2.0156 | $\mathbf{0 . 0 1 5 2}$ | $\mathbf{0 . 0 3 9 5}$ | $\mathbf{0 . 2 5}$ | $\mathbf{0 . 0 1 2 6}$ |
| HSCNN-R        | $\mathbf{0 . 0 2 1}$ | $\mathbf{0 . 0 4 9}$ | 10.12 | 12.05 | 13.911 | 3.0145 | $\mathbf{0 . 0 1 4 3}$ | $\mathbf{0 . 0 3 1 2}$ | 3.12               | 1.024                  |



Reference

* [GitHub - ss-hyun/improving-spectral-resolution_HSCNN-R](https://github.com/ss-hyun/improving-spectral-resolution_HSCNN-R)

* [GitHub - ngchc/HSCNN-Plus: HSCNN+: Advanced CNN-Based Hyperspectral Recovery from RGB Images in CVPRW 2018 (Winner of NTIRE Challenge)](https://github.com/ngchc/HSCNN-Plus)