# Conditional Generation

To generate specific images $x$ with a given condition $y$ using our novel classifier-free diffusion model, we conducted performance tests on various datasets. 
Our method demonstrates excellent conditional generation capabilities on MNIST dataset, achieving near-perfect results.
However, when applied to Chinese font generation, our method exhibits suboptimal performance. 
We hypothesize that this discrepancy in performance may be attributed to the utilization of a poorly pretrained classifier neural network.
Despite the challenges encountered in our conditional generation of Chinese font images, we have achieved superior results compared to the generic diffusion model generation.

Generation of number $8$:

<img decoding="async" src="https://github.com/morph42/conditional_generation/blob/main/number_generation/imgs/sample.png" width="50%">

Generation of Chinese font "苏体" with classifier:

<img decoding="async" src="https://github.com/morph42/conditional_generation/blob/main/chinese_font_generation/imgs/cond_img.png" width="50%">

and without classifier:

<img decoding="async" src="https://github.com/morph42/conditional_generation/blob/main/chinese_font_generation/imgs/generic_img.png" width="50%">
