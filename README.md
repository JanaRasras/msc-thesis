## Thesis 
Welcome to the GitHub repository for my thesis project. This repository contains the code and resources related to my thesis titled "Moving Sound Sources Direction of Arrival Classification Using Different Deep Learning Schemes."

### Abstract
Sound source localization is an important task for several applications and the use of deep learning for this task has recently become a popular research topic. While the majority of the previous work has focused on static sound sources, in this work we evaluate the performance of a deep learning classification system for localization of high-speed moving sound sources. In particular, we systematically evaluate the effect of a wide range of parameters at three levels including: data generation (e.g., acoustic conditions), feature extraction (e.g., STFT parameters), and model training (e.g., neural network architectures). We evaluate the performance of multiple metrics in terms of precision, recall, F-score and confusion matrix in a multi-class multi-label classification framework. We used four different deep learning models: feedforward neural networks, recurrent neural network, gated recurrent networks and temporal Convolutional neural network. We showed that (1) the presence of some reverberation in the training dataset can help in achieving better detection for the direction of arrival of acoustic sources, (2) window size does not affect the performance of static sources but highly affects the performance of moving sources, (3) sequence length has a significant effect on the performance of recurrent neural network architectures, (4) temporal convolutional neural networks can outperform both recurrent and feedforward networks for moving sound sources, (5) training and testing on white noise is easier for the network than training on speech data, and (6) increasing the number of elements in the microphone array improves the performance of the direction of arrival estimation.


### Full Thesis Document
The full thesis document can be accessed [here](https://ruor.uottawa.ca/handle/10393/44824). It includes a detailed explanation of the research, methodology, and findings.


### Citation

If you wish to reference this work, please use the following citation:
1. https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Lb-zsiIAAAAJ&sortby=pubdate&citation_for_view=Lb-zsiIAAAAJ:0KyAp5RtaNEC
2. https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Lb-zsiIAAAAJ&sortby=pubdate&citation_for_view=Lb-zsiIAAAAJ:uWiczbcajpAC
