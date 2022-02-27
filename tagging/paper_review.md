# Paper Review

---

### 1. End-to-End Object Detection with Transformers

#### Abstract

We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed compo- nents like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bi- partite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the re- lations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time perfor- mance on par with the well-established and highly-optimized Faster R- CNN baseline on the challenging COCO object detection dataset. More- over, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms com- petitive baselines. Training code and pretrained models are available at https://github.com/facebookresearch/detr.

> **요약** :
DETR 방법에 관한 논문으로써 기존의 Object Detection 과 달리 특별한 라이브러리가 필요 없고 간단한 아키텍처를 제시함

#### tagging

`CV`, `DETR`, `COCO object detection dataset`, `fast R-CNN`, `CVPR`

#### paper link

https://link.springer.com/chapter/10.1007/978-3-030-58452-8_13


---

### 2. UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild

#### Abstract

We introduce UCF101 which is currently the largest dataset of human actions. It consists of 101 action classes, over 13k clips and 27 hours of video data. The database consists of realistic user-uploaded videos containing cam- era motion and cluttered background. Additionally, we pro- vide baseline action recognition results on this new dataset using standard bag of words approach with overall perfor- mance of 44.5%. To the best of our knowledge, UCF101 is currently the most challenging dataset of actions due to its large number of classes, large number of clips and also unconstrained nature of such clips.

> **요약** :
실제 카메라로 찍은 영상들 즉 흔들림이나 배경의 이동 등등 사용자가 업로드한 비디오들로 이루어진 DataSet 으로 현재까지 가장 까다로운 dataset 으로 평가된다.

#### tagging

`DataSet`, `Video`, `user upload`

#### paper link

https://arxiv.org/abs/1212.0402

---

### 3. VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

#### Abstract

In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3 × 3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisa- tion and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facili- tate further research on the use of deep visual representations in computer vision.

> **요약** :
Convolution Network 에서 그 깊이가 정확도에 미치는 영향에 대해 설명한다.
depth를 16 - 19 정도로 깊이를 두었다.

#### tagging

`CNN`, `very deep CNN`, `ICLR`

#### paper link

https://arxiv.org/abs/1409.1556

---

### 4. DenseBox: Unifying Landmark Localization with End to End Object Detection

#### Abstract

How can a single fully convolutional neural network (FCN) perform on object detection? We introduce DenseBox, a unified end-to-end FCN framework that directly predicts bounding boxes and object class confidences through all locations and scales of an image. Our contribution is two-fold. First, we show that a single FCN, if designed and optimized carefully, can detect multiple different objects extremely accurately and efficiently. Second, we show that when incorporating with landmark localization during multi-task learning, DenseBox further improves object detection accuray. We present experimental results on public benchmark datasets including MALF face detection and KITTI car detection, that indicate our DenseBox is the state-of-the-art system for detecting challenging objects such as faces and cars.

> **요약** :
Single Fully CNN 에서 좋은 설계와 최적화를 통해 효율적으로 감지함
multi-task learning 에서 landmark 의 위치파악가 통합할 때 정확도를 더욱 향상시킴
까다로운 물체를 감별할 때 더욱 좋은 효과를 보임

#### tagging

`End to End Object Detection`, `DenseBox`, `KITTI`, `CNN`, `FCN`

#### paper link

https://arxiv.org/abs/1509.04874

---

### 5. SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE

#### Abstract

Recent research on deep convolutional neural networks (CNNs) has focused pri- marily on improving accuracy. For a given accuracy level, it is typically possi- ble to identify multiple CNN architectures that achieve that accuracy level. With equivalent accuracy, smaller CNN architectures offer at least three advantages: (1) Smaller CNNs require less communication across servers during distributed train- ing. (2) Smaller CNNs require less bandwidth to export a new model from the cloud to an autonomous car. (3) Smaller CNNs are more feasible to deploy on FP- GAs and other hardware with limited memory. To provide all of these advantages, we propose a small CNN architecture called SqueezeNet. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters. Additionally, with model compression techniques, we are able to compress SqueezeNet to less than 0.5MB (510× smaller than AlexNet).
The SqueezeNet architecture is available for download here: https://github.com/DeepScale/SqueezeNet

> **요약** :
CNN 에서 작은 모델은 3가지 이점을 제공한다
첫째 : 분산 학습중 서버간 통신이 적음
둘째 : 자율 자동차로 모델을 내보내는데 대역폭이 덜 필요하다.
셋째 : FP-GA 등 다른 하드웨어에 배포가 용이하다.
이러한 작은 모델을 위해 squeezenet 을 제안한다. 이 모델은 ImageNet 에 비해 50배 적은 매개변수로 AlexNet 수준의 정확도를 보인다.

#### tagging

`CNN`, `SqueezeNet`, `AlexNet`, `ImageNet`, `ICLR`

#### paper link

https://arxiv.org/pdf/1602.07360.pdf

---

### 6. FRACTALNET: ULTRA-DEEP NEURAL NETWORKS WITHOUT RESIDUALS

#### Abstract

We introduce a design strategy for neural network macro-architecture based on self- similarity. Repeated application of a simple expansion rule generates deep networks whose structural layouts are precisely truncated fractals. These networks contain interacting subpaths of different lengths, but do not include any pass-through or residual connections; every internal signal is transformed by a filter and nonlinearity before being seen by subsequent layers. In experiments, fractal networks match the excellent performance of standard residual networks on both CIFAR and ImageNet classification tasks, thereby demonstrating that residual representations may not be fundamental to the success of extremely deep convolutional neural networks. Rather, the key may be the ability to transition, during training, from effectively shallow to deep. We note similarities with student-teacher behavior and develop drop-path, a natural extension of dropout, to regularize co-adaptation of subpaths in fractal architectures. Such regularization allows extraction of high- performance fixed-depth subnetworks. Additionally, fractal networks exhibit an anytime property: shallow subnetworks provide a quick answer, while deeper subnetworks, with higher latency, provide a more accurate answer.

> **요약** :
자기 유사성을 기반으로 한 neural network 매크로 아키텍처 설계를 소개한다.


#### tagging

#### paper link

https://arxiv.org/abs/1605.07648

---

### 7.

#### Abstract

#### tagging

#### paper link

---

### 8.

#### Abstract

#### tagging

#### paper link

---

### 9.

#### Abstract

#### tagging

#### paper link

---

### 10.

#### Abstract

#### tagging

#### paper link

---

### 11.

#### Abstract

#### tagging

#### paper link

---

### 12.

#### Abstract

#### tagging

#### paper link

---

### 13.

#### Abstract

#### tagging

#### paper link

---

### 14.

#### Abstract

#### tagging

#### paper link

---

### 14.

#### Abstract

#### tagging

#### paper link

---

### 14.

#### Abstract

#### tagging

#### paper link

---

### 14.

#### Abstract

#### tagging

#### paper link

---

### 14.

#### Abstract

#### tagging

#### paper link

---

### 14.

#### Abstract

#### tagging

#### paper link

---

### 14.

#### Abstract

#### tagging

#### paper link
