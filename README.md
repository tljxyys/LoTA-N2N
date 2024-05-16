# Low-Trace Adaptation of Zero-shot Self-supervised Blind Image Denoising🚀
[![](https://img.shields.io/badge/Project-Page-green.svg)](https://github.com/tljxyys/LoTA-N2N) [![](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2403.12382) [![](https://img.shields.io/badge/Dataset-🔰Kodak24-blue.svg)](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset) [![](https://img.shields.io/badge/Dataset-🔰McMaster18-blue.svg)](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm) [![](https://img.shields.io/badge/Dataset-🔰FMD-blue.svg)](https://paperswithcode.com/dataset/fmd) [![](https://img.shields.io/badge/Dataset-🔰Xray-blue.svg)](https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct) 


***
>**Abstract**: _Deep learning-based denoiser has been the focus of recent development on image denoising. In the past few years, there has been increasing interest in developing self-supervised denoising networks that only require noisy images, without the need for clean ground truth for training. However, a performance gap remains between current self-supervised methods and their supervised counterparts. Additionally, these methods commonly depend on assumptions about noise characteristics, thereby constraining their applicability in real-world scenarios. Inspired by the properties of the Frobenius norm expansion, we discover that incorporating a trace term reduces the optimization goal disparity between self-supervised and supervised methods, thereby enhancing the performance of self-supervised learning. To exploit this insight, we propose a trace-constraint loss function and design the low-trace adaptation Noise2Noise (LoTA-N2N) model that bridges the gap between self-supervised and supervised learning. Furthermore, we have discovered that several existing self-supervised denoising frameworks naturally fall within the proposed trace-constraint loss as subcases. Extensive experiments conducted on natural and confocal image datasets indicate that our method achieves state-of-the-art performance within the realm of zero-shot self-supervised image denoising approaches, without relying on any assumptions regarding the noise._
>

![image](https://github.com/tljxyys/LoTA-N2N/blob/main/fig/Architecture.png)
***

## 1. Background
Denoising refers to the process of removing noise from data, typically within the context of image processing. Noise in an image can stem from various sources, such as suboptimal lighting conditions, sensor imperfections, or transmission inconsistencies. Within the realm of deep learning, denoising involves training neural networks to discern the inherent structure of the noisy data, enabling them to predict a clean, noise-free version of the input. Mathematically, denoising aims to approximate a function $f_{\theta}(\cdot)$, parameterized by $\theta$, which maps a noisy input $y$ to a corresonding clean output $x$: 

$$fθ(y) ≈ x \tag{1}$$

Denoising methodologies can be classified into two categories based on the nature of training data: supervised and self-supervised (unsupervised). Supervised denoising requires pairs of clean and noisy data for training. The denoising function uses noisy inputs to produce denoised outputs, which are then compared to the clean data to minimize discrepancies. Such methods benefit from the direct learning signals provided by paired data, promoting a more precise understanding of the noise-to-signal mapping. In contrast, self-supervised denoising does not require labeled datasets. Instead, it aims to infer a clean data representation directly from the noisy inputs by optimizing an objective function. This function compels the network to learn the inherent structure of the data and filter out the noise. Self-supervised methods are based on the assumption that clean data reside within a lower-dimensional manifold of the noisy input space, which can be leveraged to dissociate the signal from the noise. While these pioneering techniques have advanced self-supervised denoising, they frequently rest upon assumptions about the noise characteristics that may not be valid in complex real-world contexts. This limitation often leads to suboptimal performance when these methods are applied to data with unanticipated noise distributions. Therefore, there is a clear need for denoising approaches that do not rely on any predefined assumptions about noise. 

## 2. Main Idea

Figurea above illustrates the workflow of mainstream denoising algorithms in comparison to the process diagram of our proposed LoTA-N2N (Low-trace Adaptation Noise2Noise) model. Supervised denoising is trained using pairs of clean and noisy images. The Noise2Noise approach replaces these pairs with noisy/noisy image pairs, achieving denoising without the need for clean samples. Neighbor2Neighbor further reduces dataset requirements by generating noisy/noisy image pairs through downsampling a single noisy image. Our method, LoTA-N2N, takes a further step in the loss function by constraining the trace term. It guides the self-supervised model closer to the direction of supervised learning and yields superior performance without necessitating any prior assumptions about the noise characteristics.

In this work, we address the inherent shortcomings of conventional self-supervised denoising models that utilize the Noise2Noise (N2N) framework, which relies on mean squared error (MSE) loss for training. Since the noisy sub-images produced by the downsampling process do not conform to the assumption of equal mean intensities, directly applying the MSE loss leads to biased estimates in the trained models. To overcome this challenge, we propose a decomposition of the MSE loss, dividing it into terms suitable for a supervised learning framework, plus an additional trace component. This methodology is expected to improve the performance of denoising networks by providing an effective strategy for more precise noise reduction in practical applications.

## 3. Results

![image](https://github.com/tljxyys/LoTA-N2N/blob/main/fig/Result1.png)

![image](https://github.com/tljxyys/LoTA-N2N/blob/main/fig/Result2.png)

## 4. Conclusion

In this paper, we propose a novel trace-constraint loss function that bridges the gap between self-supervised and supervised learning in the field of image denoising. By effectively optimizing the self-supervised denoising objective through the incorporation of a trace term as a constraint, our approach allows for improved performance and generalization across various types of images including natural, medical, and biological imagery. We enhance the designed trace-constraint loss function by incorporating the concepts of mutual study and residual study to achieve improved denoising performance and generalization. Furthermore, our designed model has been kept lightweight, enabling better denoising results to be achieved in a shorter training time, without the need for any prior assumptions about the noise. Our method outperforms existing self-supervised denoising models by a significant margin, demonstrating its potential for widespread adoption and practicality in real-world scenarios. Overall, our approach represents a valuable contribution to the advancement of self-supervised denoising methods and holds promise for addressing practical challenges associated with acquiring paired clean / noisy images for supervised learning.

## Bibtex
```
@misc{hu2024lowtrace,
      title={Low-Trace Adaptation of Zero-shot Self-supervised Blind Image Denoising}, 
      author={Jintong Hu and Bin Xia and Bingchen Li and Wenming Yang},
      year={2024},
      eprint={2403.12382},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

