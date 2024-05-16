# Low-Trace Adaptation of Zero-shot Self-supervised Blind Image Denoising
[![](https://img.shields.io/badge/Project-Page-green.svg)](https://github.com/tljxyys/LoTA-N2N) [![](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2403.12382)


***
>**Abstract**: _Deep learning-based denoiser has been the focus of recent development on image denoising. In the past few years, there has been increasing interest in developing self-supervised denoising networks that only require noisy images, without the need for clean ground truth for training. However, a performance gap remains between current self-supervised methods and their supervised counterparts. Additionally, these methods commonly depend on assumptions about noise characteristics, thereby constraining their applicability in real-world scenarios. Inspired by the properties of the Frobenius norm expansion, we discover that incorporating a trace term reduces the optimization goal disparity between self-supervised and supervised methods, thereby enhancing the performance of self-supervised learning. To exploit this insight, we propose a trace-constraint loss function and design the low-trace adaptation Noise2Noise (LoTA-N2N) model that bridges the gap between self-supervised and supervised learning. Furthermore, we have discovered that several existing self-supervised denoising frameworks naturally fall within the proposed trace-constraint loss as subcases. Extensive experiments conducted on natural and confocal image datasets indicate that our method achieves state-of-the-art performance within the realm of zero-shot self-supervised image denoising approaches, without relying on any assumptions regarding the noise._>

![image](https://github.com/tljxyys/LoTA-N2N/blob/main/fig/Architecture.png)
***
