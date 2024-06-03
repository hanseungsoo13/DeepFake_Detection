# DeepFake_Detection

### Summary

딥페이크 탐지 모델은 영상이 주어지면 즉각적으로 탐지하는 것이 중요하다. 따라서 본 연구는 기존의 딥페이크 탐지 알고리즘에서 딥페이크 탐지 시간을 줄이는 것을 목적으로 하며, Gray Channel, SimCLR, ConvLSTM등을 적용하여 실험을 진행하고 가장 효율적인 딥페이크 탐지 방법론을 찾기 위해 연구함.


# 1. Introduction

### DeepFake Detection이란?

- **DeepFake**
    - 딥러닝(DeepLearning)과 가짜(Fake)의 합성어로 영상에 사람의 얼굴을 합성하여 새로운 영상을 제작하는 기술
    - 최근 생성 기법인 GAN을 기반으로 하여 정교한 딥페이크 기술이 발전하고 있음
- **DeepFake Detection**
    - 딥페이크 탐지를 목적으로 하는 모델
    - 주로 합성곱 신경망(CNN)을 기반으로 발전
    

# **2. Related works**

1. Gray 채널 분석을 사용한 딥페이크 탐지 성능 비교 연구(2021)
    
    [Gray 채널 분석을 사용한 딥페이크 탐지 성능 비교 연구 | DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10612133)
    
    ![related works](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/37dae048-43f7-407b-8373-cfd40a690b42)
    
    - **Gray 채널**을 활용한 딥페이크 탐지의 효율성 확인
2. A method of Detection of Deepfake Using Bidirectical Convolutional LSTM
    
    [A Method of Detection of Deepfake Using Bidirectional Convolutional LSTM
    				-Journal of the Korea Institute of Information Security & Cryptology
    			
    		
    	 | Korea Science](https://koreascience.kr/article/JAKO202006763002809.page)
    
    - 딥페이크 탐지 모델에서 **ConvLSTM**의 적용 가능성 확인

# 3. Method

효과적이고 효율적인 DeepFake Detector를 위해 본 팀의 Main Idea는 Contrastive Learning을 통해 representation을 학습하고 이 representation 정보를 이용한 Detector 모델을 고안해 내는 것입니다.

## 3.1 Overall Flow

![overall flow](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/2a57e030-f339-4837-a897-0d9df5d794bb)

전체적인 Overall Flow는 위 그림을 통해 알 수 있듯이 크게 3가지의 과정을 거칩니다.

1. DeepFake Video에 FaceDetetor(RetinaFace)를 통해 **얼굴 부분만 crop**
2. **SimCLR**을 이용해 DeepFake 영상의 **representation 학습**
3. 학습된 **representation**을 이용한 **Classifier모델(FPN+Swin Transformer)**로 Real 영상과 Fake 영상을 분류

## 3.2 Proposed Model Architecture

### FaceDetector

Deepfake video는 사람의 얼굴 부분의 부자연스러움이 특징이므로 Video에서 얼굴 부위만 Detector로 Crop 하여 학습합니다.

FaceDetector로 **Retina Face**(2019)를 활용하였습니다.

![face detector](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/d29184ae-5d87-42b1-9035-7962fa32209c)

- **Retina Face란?**
    
    [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
    
    **Face recognition**부분에 SOTA를 달성한 다양한 방식들을 결합하여 정확도를 향상시킨 모델
    
    - **Feature Pyramid**: 컴퓨팅 자원을 적게 차지하면서 이미지 안의 다양한 크기의 객체를 인식하는 방법
    - **Single-Stage**: Two-stage에 비해 불균형한 데이터에서 얼굴의 위치와 크기를 효과적으로 잡아냄
    - **Context Modeling**: DCN(Deformable Convolution Network)을 이용하여 좀 더 flexible한 receptive field에서 특성을 추출한다.

### Representation Learning

Deep Fake 영상에 Face Detector를 적용해 얻은 Deep Fake face data들의 Representation을 학습하기 위한 과정입니다. 

이 과정에는 정답 라벨이 없기 때문에 일반적인 Supervised Learning이 아닌 **Self-Supervised Learning**을 시도했고, 여러 방법 중 **Contrastive Learning**의 대표 모델인 **SimCLR**을 이용하기로 선택하였습니다.

- **SimCLR이란?**
    
    [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
    
    ![simclr](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/9456419d-d356-4a27-aac5-c7b07b77710e)
    
    - **Self-superviesed Learning**을 활용하여 **Contrastive learning** 기법의 학습 방식
    - 다양한 **Augmentation**을 가한 image를 학습
    - 물리적인 환경 변화에도 **Robustness**를 유지
    - FC layer를 Projection head로 교체
    - 자세한 리뷰는 아래 논문 리뷰 참고해주세요!
        
        [A Simple Framework for Contrastive Learning of Visual Representation](https://www.notion.so/A-Simple-Framework-for-Contrastive-Learning-of-Visual-Representation-a26e79def96949d1bb40ba3cf9cf9492?pvs=21)
        
- 본 프로젝트에서는 Augmented된 Image의 representation을 학습하는 Convolution Network $f(\cdot)$에 **ResNet18**과 **ResNet50**으로 실험하였습니다.
- Augmentation 과정에는 논문에서 제시한대로 **Crop, Gaussian blur, Horizontal flip, Color Distortion**을 적용하였다.
- 학습된 Representation은 Convolution Network의 가중치를 통해 전달합니다.

### Classifier

SimCLR을 통해 학습된 ResNet의 가중치를 이용하여 **실제 영상과 DeepFake 영상을 분류**하는 과정입니다. 이 과정에서는 **ResNet을 Backbone**으로 하면서 **Video data** 학습이 가능한 Vision & Time Series 모델이 필요했습니다. 

당시 프로젝트에는 위의 Related Work에 있는 논문을 참고하여 **ConvLSTM(2015)**을 Classifier로 활용했지만 프로젝트를 다시 정리하는 과정에서 추가적인 성능 향상을 위해 **FPN(2016)**과 **Video-Swin Transformer(2021)**를 활용하였습니다.

- **Conv-LSTM이란?**
    
    ![convlstm](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/6ece820e-5dc2-47ab-8edb-7e6bb5219deb)
    
    - 시간의 흐름에 따라 **Convolution** 연산을 수행하는 방식
    - Convolustion 연산을 통해 **공간적 특징**을, **LSTM**을 통해 **시간적 특징**을 학습할 수가 있다.
- **FPN & Video-Swin Transformer 설명**
    1. **FPN(Feature Pyramid Network)**
        
        ![fpn](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/59090c3f-bd53-4665-97b7-4edc9671fa94)
        
        - **Feature map을 Pyramid처럼 쌓아 올린 구조**로 다양한 scale의 Feature-map을 활용하기 때문에 다양한 scale에서 Object Detection이 가능하다는 장점이 있습니다.
        - **Buttom-Up** 방식으로 Feature-map의 크기를 점점 줄여가며 학습한 뒤에 **Top-Down** 방식으로 Feature-map의 크기를 다시 키워가면서 Detection하는 과정입니다.
        - 모델에 대한 자세한 리뷰는 아래 참고해주시면 됩니다.
            
            [Feature Pyramid Networks for Object Detection](https://www.notion.so/Feature-Pyramid-Networks-for-Object-Detection-642f1e1a810a474499eeba4cb7b900e1?pvs=21)
            
    2. Video Swin Transformer
        
        ![video swin transformer](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/4fe351ad-5096-461e-93a3-3ae07e7ac5e7)
        
        - Image understaning 분야에서 개발된 모델인 **Swin Transformer**의 핵심 아이디어인 hierarchical structure에 local attention의 범위를spatiotemporal domain까지 확장하면서 **video task**에서 좋은 성능을 보인 모델
        - 논문 리뷰 참고
            
            [[논문 리뷰] Video Swin Transformer](https://healess.github.io/paper/Paper-Video-Swin-Transformer/)
            
- Conv-LSTM은 ResNet의 feature map을 Input으로 받아 이어서 학습
- FPN은 Buttom-Up Network를 ResNet으로 구성하여 앞서 학습했던 ResNet의 가중치들을 그대로 옮겨와서 학습

## 3.3 Dataset

최근 딥페이크 탐지 연구에서 주로 사용되는 Dataset 선별

- Celeb-DF
- FF++
- DFDC

![dataset](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/9f252013-730f-416c-bb0f-78aabc4d9d9e)
# 4. Experiments

### 1. 정확도

![experiment1](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/852c77d1-c63c-4dde-aa57-4000eb794007)

- Celeb-DF는 Gray Channel의 성능이, FF++데이터셋은 RGB의 성능이 좋았다.
- ResNet18과 ResNet50 역시 데이터셋에 따라서 성능이 달라졌다.

### 2. Time Complexity

![experiment2](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/532cfb74-7b89-4dc1-94cb-17a77ff4f076)

- **ResNet18을 사용한 모델이 예측 시간이 많이 단축**되었다.
- RGB채널과 Gray채널로 변환한 데이터 간의 예측 시간에는 큰 차이가 없었다.

### 3. ConvLSTM

![experiment3](https://github.com/hanseungsoo13/DeepFake_Detection/assets/75753717/5db14d91-ee90-4244-b47d-d356db1b1a80)

- LSTM과 **Conv-LSTM**에서 정확도에는 확연한 차이가 있다.

# 5. Conclusion

- Self-Supervised Learning을 통해 다양한 상황에서도 Robust한 detection 성능 확보
    - 특히 적은 파라미터 수를 가진 ResNet18과 ResNet50으로 representation learning을 했음에도 불구하고 좋은 성능을 보임
- ConvLSTM, Video Swin Transformer 등과 같이 Classifier에 공간적, 시간적 특징을 반영시킴으로써 성능 향상 유도
- 무의미한 색상정보를 최소화 하기 위해 Video를 Gray Channel로 전처리하여 효율성 확보
