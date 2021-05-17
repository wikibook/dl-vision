**> 4장:**
<a href="https://www.packtpub.com" title="Get the book!">
    <img src="../banner_images/book_cover.png" width=200 align="right">
</a>
# 유력한 분류 모델

4장은 _VGG_, _Inception_, _ResNet_ 같은 유력한 CNN 아키텍처를 다루고 일반적으로 컴퓨터 비전과 머신러닝에 기여한 바가 무엇인지 알아본다. 더 복잡한 분류 작업을 소개하면서, CNN이 다양한 데이터셋에서 얻은 지식을 어떻게 활용할 수 있는지(_전이 학습_) 설명한다. 다음 노트북에서는 이러한 유력한 모델 중 몇 가지를 자세히 구현하고, 다양한 플랫폼에서 공유하고 있는 사전에 구현되고 훈련된 모델이 어떻게 효율적으로 재사용될 수 있는지 보여준다. 

## :notebook: 노트북

(팁: 노트북을 시각화할 때 `nbviewer`를 사용하는 것이 좋다: `nbviewer.jupyter.org`에서 계속하려면 [여기](https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-Tensorflow/blob/master/ch4)를 클릭하라.)

- 4.1 - [ResNet을 처음부터 구현하기](./ch4_nb1_implement_resnet_from_scratch.ipynb)
    - 상당히 깊은 _ResNet_ 아키텍처 (_ResNet-18_, _ResNet-50_, _ResNet-152_)를 블록 단위로 구현하고 `tensorflow-datasets`를 통해 얻은 큰 데이터셋 (_CIFAR-100_)의 분류에 적용한다. 
- 4.2 - [케라스 애플리케이션의 모델 재사용하기](./ch4_nb2_reuse_models_from_keras_apps.ipynb)
    - `keras.applications`에서 얻을 수 있는 사전 구현된 모델을 재사용하고 _ResNet-50_의 다른 버전을 훈련시키는 방법을 알아본다.
- 4.3 - [텐서플로 허브에서 모델 가져오기](./ch4_nb3_fetch_models_from_tf_hub.ipynb)
    - [tfhub.dev](http://tfhub.dev)에서 사전 훈련된 모델에 대한 온라인 카탈로그를 탐색하고, `tensorflow-hub`를 사용해 사전 훈련된 모델을 가져와 인스턴스화한다(바로 사용할 수 있게 준비된  _Inception_과 _MobileNet_ 모델을 실험한다). 
- 4.4 - [전이 학습 적용하기](./ch4_nb4_apply_transfer_learning.ipynb)
    - 다양한 데이터셋에서 사전 훈련된 모델을 미세하게 조정하거나 고정해보면서 _전이 학습_을 실험한다. 
- 4.5 - (Appendix) [ImageNet과 Tiny-ImageNet 살펴보기](./ch4_nb5_explore_imagenet_and_its_tiny_version.ipynb)
    - _ImageNet_과 _Tiny-ImageNet_에 대해 더 자세히 알아보고, 더 복잡한 데이터셋에서 모델을 훈련시키는 방법을 배운다.
	
## :page_facing_up: 추가 파일

- [cifar_utils.py](cifar_utils.py): `tensorflow-datasets`을 사용한, _CIFAR_ 데이터셋을 위한 유틸리티 함수 (코드: 노트북 [4.1](./ch4_nb1_implement_resnet_from_scratch.ipynb))
- [classification_utils.py](classification_utils.py): 분류 작업(이미지를 로드하거나 예측을 표시하는 등)을 위한 유틸리티 함수 (코드: 노트북 [4.1] (./ch4_nb1_implement_resnet_from_scratch.ipynb)).
- [keras_custom_callbacks.py](keras_custom_callbacks.py): 모델 훈련 과정을 모니터링하기 위한 맞춤형 케라스 _콜백 함수_ (코드: 노트북 [4.1](./ch4_nb1_implement_resnet_from_scratch.ipynb)).
- [resnet_functional.py](resnet_functional.py): _케라스 함수형_ API를 사용해 _ResNet_을 구현 (코드: 노트북 [4.1](./ch4_nb1_implement_resnet_from_scratch.ipynb)). 
- [resnet_objectoriented.py](resnet_objectoriented.py): _케라스 함수형_ API를 사용해 _ResNet_을 구현하되, 이번에는 _객체 지향_ 패러다임을 따라 구현함 (코드: 노트북 [4.1](./ch4_nb1_implement_resnet_from_scratch.ipynb)).
- [tiny_imagenet_utils.py](tiny_imagenet_utils.py): _Tiny-ImageNet_ 데이터셋을 위한 유틸리티 함수(코드: 노트북 [4.5](./ch4_nb5_explore_imagenet_and_its_tiny_version.ipynb)).
