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

## 2.1. Revisit of other methods

The effectiveness of our proposed LoTA-N2N model can be theoretically supported. The discrepancy between self-supervised learning and supervised learning is attributable to their distinct optimization objectives. In our proposed method, we suggest that the loss function in self-supervised learning can be decomposed into the supervised learning loss component and an additional term. By minimizing this additional term towards zero, we can potentially align the convergence of self-supervised learning with that of supervised learning, thus achieving significant performance gains in self-supervised denoising models. To demonstrate this decomposition, we introduce the following lemmas.

**Lemma 1.** Given a matrix $\mathbf{A}\in\mathbb{R}^{n \times n}$, the following identity holds:

$$ \Vert\mathbf{A}\Vert^{2}_{2}=Tr(\mathbf{A}^\text{T}\mathbf{A}) \tag{2}$$

where $\Vert \cdot \Vert^{2}_{2}$ denotes the Frobenius norm (element-wise 2-norm), summed across all squared elements of the matrix, and $Tr(\cdot)$ is the trace operation of a matrix.

**Lemma 2.** For any two matrices $\mathbf{A}$, $\mathbf{B}\in\mathbb{R}^{n \times n}$, we have:

$$ \Vert\mathbf{A}\pm\mathbf{B}\Vert{^2_2} = \Vert\mathbf{A}\Vert{^2_2} + \Vert\mathbf{B}\Vert{^2_2}\pm 2Tr (\mathbf{A}^\text{T}\mathbf{B}) \tag{3}$$

Using Lemma 2, we can restructure the loss of self-supervised approach as the loss in supervised learning plus or minus a trace term and a constant. The disparity between the results of self-supervised and supervised learning arises primarily from the behavior of this trace term. A logical approach might involve setting this trace term to zero, thereby bridging the gap between the performance of self-supervised and supervised learning, leading to considerable improvements in performance. In light of this, we review several prominent self-supervised denoising models.

**Revisit Noise2Noise:** Noise2Noise was a pioneering approach among self-supervised denoising methods. Instead of using noisy/clean image pairs, Noise2Noise leveraged noisy/noisy image pairs with mutually independent noise. Specifically, the pairs of noisy images in Noise2Noise can be described as follows:
$$y=x+n,\quad n \sim \mathcal{N}\left(\textbf{0}, \sigma{^2_1}\textbf{\textit{I}}\right),\quad y'=x+n',\quad n \sim \mathcal{N}\left(\textbf{0}, \sigma{^2_2}\textbf{\textit{I}}\right) \tag{4}$$

where $y$ and $y'$ constitute two independent noisy representations of a clean image $x$. Utilizing Lemma 2, the optimization objective of Noise2Noise can be reformulated:
$$
\begin{aligned}
    &\quad\ \ \mathcal{L}_{Noise2Noise}(\theta)\\
    &=\mathbb{E}_{n, n'}\{\Vert\mathbf{f_{\theta}(\mathbf{y})}-\mathbf{y'}\Vert^{2}_{2}\}=\mathbb{E}_{n, n'}\{\Vert\mathbf{f_{\theta}(\mathbf{y})}-\mathbf{x}-\mathbf{n'}\Vert^{2}_{2}\}\\
    &=\mathbb{E}_{n, n'}\{\Vert\mathbf{f_{\theta}(\mathbf{y})}-\mathbf{x}\Vert^{2}_{2}-2\Tr\{(\mathbf{f_{\theta}(\mathbf{y})}-\mathbf{x})^\text{T}\mathbf{n'}\}+\Vert\mathbf{n'}\Vert^{2}_{2}\}\\
    &=\mathbb{E}_{n, n'}\{\Vert\mathbf{f_{\theta}(\mathbf{y})}-\mathbf{x}\Vert^{2}_{2}\}-2\mathbb{E}_{n, n'}\{\Tr\{(\mathbf{f_{\theta}(\mathbf{y})})^\text{T}\mathbf{n'}\}\}+C\\
    &=\mathbb{E}_{n, n'}\{\Vert\mathbf{f_{\theta}(\mathbf{y})}-\mathbf{x}\Vert^{2}_{2}\}-2\mathbb{E}_{n, n'}\{\Tr\{(\mathbf{n'})^\text{T}\mathbf{f_{\theta}(\mathbf{y})}\}\}+C\\
    &=\mathbb{E}_{n, n'}\{\Vert\mathbf{f_{\theta}(\mathbf{y})}-\mathbf{x}\Vert^{2}_{2}\}-2\Tr\{\mathbb{E}_{n, n'}\{(\mathbf{n'})^\text{T}\mathbf{f_{\theta}(\mathbf{y})}\}\}+C.
\end{aligned}
$$
Here, $C$ equals $\mathbb{E}_{n, n'}\{\Vert\mathbf{x}-\mathbf{y'}\Vert^{2}_{2}-2\Tr(\mathbf{x}^\text{T}(\mathbf{n'}))\}$,  which is a constant independent of $\theta$. The notation $f_{\theta}(\cdot)$ represents the denoising network characterized by learnable parameters $\theta$.

Given the statistical independence and zero-mean nature of $n$ and $n'$, we can assert:
\begin{equation}
\begin{aligned}
    &\quad \ \mathbb{E}_{n, n'}\{(\mathbf{n'})^\text{T}\mathbf{f_{\theta}(\mathbf{y})}\}\\
    &=Cov_{n, n'}((\mathbf{n'})^\text{T},\ \mathbf{f_{\theta}(\mathbf{y})})
    =Cov_{n, n'}(\boldsymbol{\sigma}\mathbf{n'}, \mathbf{M}\mathbf{y}+\mathbf{N})\\
    &=Cov_{n, n'}(\boldsymbol{\sigma}\mathbf{n'}, \mathbf{M}\mathbf{n})=\boldsymbol{\sigma}Cov\left(\mathbf{n'},\mathbf{n}\right)\mathbf{M}^\text{T}=\textbf{0}.
\end{aligned}    
\end{equation}

Accordingly, the optimization target of N2N \cite{lehtinen2018noise2noise} becomes analogous to that of supervised training, which explains why N2N achieves performance equalling or closely approaching its supervised counterparts. The proof also indicates that once $\mathbf{n}$ and $\mathbf{n}^{\prime}$ are confirmed to be mutually independent, the trace term nullifies, allowing self-supervised learning to mimic the properties of supervised learning. \\

\noindent
\textbf{Revisit Noisy As Clean:} The Noisy As Clean (NAC) \cite{Xu_2020} method posits that noise present in images is sufficiently subtle, facilitating training on a noisier/noise dataset. The method defines the noisier sample as $\mathbf{z}=\mathbf{x}+\mathbf{n}+\mathbf{m}$, and the noisy sample as $\mathbf{y}=\mathbf{x}+\mathbf{n}$, where 
$\mathbf{x}$ represents the clean image, $\mathbf{n}$ the observed noise, and $\mathbf{m}$ the simulated noise. The variances and expectations of both observed and simulated noise are presumed to be negligible. Echoing the Noise2Noise framework, the optimization objective of Noisy As Clean can be reformulated as:
\begin{equation}
\begin{aligned}
    &\quad\ \ \mathcal{L}_{Noisy As Clean}(\theta)\\
    &=\mathbb{E}_{n, m}\{\Vert\mathbf{f_{\theta}(\mathbf{z})}-\mathbf{y}\Vert^{2}_{2}\}=\mathbb{E}_{n, m}\{\Vert\mathbf{f_{\theta}(\mathbf{z})}-\mathbf{x}-\mathbf{n}\Vert^{2}_{2}\}\\
    &=\mathbb{E}_{n, m}\{\Vert\mathbf{f_{\theta}(\mathbf{y})}-\mathbf{x}\Vert^{2}_{2}\}-2\Tr\{\mathbb{E}_{n, m}\{(\mathbf{n})^\text{T}\mathbf{f_{\theta}(\mathbf{z})}\}\}+C.
\end{aligned}
\end{equation}

Here, $C$ is a constant term not dependent on $\theta$. The variables retain their meanings as defined in the previous section. Subsequently, we demonstrate that, under NAC's assumptions, the trace term is reduced to zero, illustrating how the optimization objective aligns with the supervised paradigm.
\begin{equation}
\begin{aligned}
    &\quad \ \mathbb{E}_{n, m}\{(\mathbf{n})^\text{T}\mathbf{f_{\theta}(\mathbf{z})}\}\\
    &=Cov_{n, m}((\mathbf{n})^\text{T},\ \mathbf{f_{\theta}(\mathbf{z})})+\mathbb{E}_{n, m}\{(\mathbf{n})^\text{T}\}\mathbb{E}_{n, m}\{\mathbf{f_{\theta}(\mathbf{z})}\}\\
    &\approx Cov_{n, m}((\mathbf{n})^\text{T},\ \mathbf{f_{\theta}(\mathbf{z})})=Cov_{n, m}(\boldsymbol{\sigma}\mathbf{n}, \mathbf{M}\mathbf{z}+\mathbf{N})\\
    &=Cov_{n, m}(\boldsymbol{\sigma}\mathbf{n}, \mathbf{M}\mathbf{n}+\mathbf{M}\mathbf{m})\\
    &=\boldsymbol{\sigma}Var\left(\mathbf{n}\right)\mathbf{M}^\text{T}+\boldsymbol{\sigma}Cov\left(\mathbf{n},\mathbf{m}\right)\mathbf{M}^\text{T}\\
    &\approx\boldsymbol{\sigma}\left(\rho_{n,m}\sqrt{Var(\textbf{n})}\sqrt{Var(\textbf{m})}\right)\mathbf{M}^\text{T}\\
    &\approx\textbf{0}.
\end{aligned}    
\end{equation}

\begin{figure*}
  \centering
  \includegraphics[width=\textwidth]{Figure_3.pdf}
  \caption{The main pipeline of our proposed method. The two-stage model begins with a pretraining phase where the network is initially trained using an MSE loss, leading to a biased denoiser. To improve performance, the subsequent fine-tuning stage employs the trace-constrained loss that supplements the model's training beyond the MSE baseline. This two-step training process aims to narrow the gap between self-supervised and supervised learning techniques, thus enhancing the overall effectiveness of the model.}
  \label{Figure 3}
\end{figure*}

Given this result, during the optimization process, the parameters' update direction, when applying the loss function derivative with respect to $\theta$, consistently coincides with that of a supervised learning setting. \\

\noindent
\textbf{Revisit Recorrupted2Recorrupted:} Rec2Rec \cite{9577798} generates pairs of data, $\mathbf{\widehat{y}}$ and $\mathbf{\widetilde{y}}$, both with independent noise from an initial noisy image $\mathbf{y}$. A neural network is then trained to map $\mathbf{\widehat{y}}$ to $\mathbf{\widetilde{y}}$. More formally:
\begin{equation}
    \mathbf{y}=\mathbf{x}+\mathbf{n},\quad \mathbf{n}\sim\mathcal{N}\left(\textbf{0}, \sigma^{2}\textbf{\textit{I}}\right),
\end{equation}
\begin{equation}
\mathbf{\widehat{y}}=\mathbf{y}+\mathbf{D}^\text{T}\mathbf{m},\quad\mathbf{\widetilde{y}}=\mathbf{y}-\mathbf{D}^{-1}\mathbf{m},\quad\mathbf{m}\sim\mathcal{N}\left(\textbf{0}, \sigma^{2}\textbf{\textit{I}}\right).
\end{equation}

We can establish that the trace term in the loss function of Recorrupted2Recorrupted is given by:
\begin{equation}
\begin{aligned}
    \Tr\{\mathbb{E}_{n, m}\{(\mathbf{f_{\theta}(\mathbf{\widehat{y}})})^\text{T}(\mathbf{n}-\mathbf{D}^{-1}\mathbf{m})\}\}.
\end{aligned}
\end{equation}

For simplicity, one may denote $\mathbf{\widehat{n}}=\mathbf{n}+\mathbf{D}^\text{T}\mathbf{m}$, $\mathbf{\widetilde{n}}=\mathbf{n}-\mathbf{D}^\text{T}\mathbf{m}$. The trace term can thus be rewritten as:
\begin{equation}
\begin{aligned}
    \Tr\{\mathbb{E}_{n, m}\{(\mathbf{f_{\theta}(\mathbf{x}+\mathbf{\widehat{n}})})^\text{T}\mathbf{\widetilde{n}}\}\}.
\end{aligned}
\end{equation}

Under the construction, $\mathbf{\widehat{n}}$ and $\mathbf{\widetilde{n}}$ are mutually independent, adhering to the condition discussed in the preceding Noise2Noise section. Similarly, it can be demonstrated that the trace term vanishes.


## Results

![image](https://github.com/tljxyys/LoTA-N2N/blob/main/fig/Result1.png)

![image](https://github.com/tljxyys/LoTA-N2N/blob/main/fig/Result2.png)

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

