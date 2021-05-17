**> 6장:**
<a href="https://www.packtpub.com" title="Get the book!">
    <img src="../banner_images/book_cover.png" width=200 align="right">
</a>
# 이미지 보강 및 분할

합성곱 신경망은 다차원 데이터를 출력하기 위해 만들어질 수 있다. 따라서 이 모델은 _이미지를 예측하기_ 위해 훈련될 수 있다. 6장에서 _인코더-디코더_와 더 구체적인 _오토인코더_를 소개하고 훼손된 이미지를 복원하거나 픽셀 단위로 분류(즉, 의미론적 분할)하기 위해 이 모델을 적용하는 방법을 보여준다. 단순한 숫자 이미지부터 자율주행 자동차 애플리케이션에서 수집된 사진까지, 다음 노트북은 CNN이 데이터를 편집하고 분할하는 방법을 설명한다. 

## :notebook: 노트북

(팁: 노트북을 시각화할 때 `nbviewer`를 사용하는 것이 좋다: `nbviewer.jupyter.org`에서 계속하려면 [여기](https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-Tensorflow/blob/master/ch6)를 클릭하라.)

- 6.1 - [오토인코더 알아보기](./ch6_nb1_discover_autoencoders.ipynb)
    - 간단한 _오토인코더(auto-encoder, AE)를 만들고 잠재 공간(_데이터 임베딩_)을 살펴본다. 
- 6.2 - [오토인코더로 노이즈 제거하기](./ch6_nb2_denoise_with_autoencoders.ipynb)
    - 앞에서 본 _오토인코더_를 훈련시켜 훼손된 이미지의 노이즈를 제거한다. 
- 6.3 - [심층 오토인코더로 이미지 품질 개선하기(초해상도)](./ch6_nb3_improve_image_quality_with_dae.ipynb)
    - 간단한 _합성곱 오토인코더_를 구현하고 이어 더 복잡한 _U-Net_을 구현한 다음 이 모델을 해상도가 낮은 미지의 해상도를 높이는 데 적용한다. 
- 6.4 - [자율주행 자동차 애플리케이션을 위한 데이터 준비하기](./ch6_nb4_preparing_data_for_smart_car_apps.ipynb)
    - 자율 주행을 위한 인식 시스템 훈련에 사용되는 유명한 데이터셋인 _Cityscapes_를 찾고 준비한다. 
- 6.5 - [의미론적 분할을 위한 FCN-8s 모델을 만들고 훈련시키기](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)
    - _VGG_를 의미론적 분할에 효율적인 모델인 _FCN-8s_로 확장하고 자율주행을 위한 시각 데이터 인식에 적용한다. _손실 측정/비교_ 전략 또한 보여준다.
- 6.6 - [객체와 인스턴스 분할을 위한 U-Net 모델을 만들고 훈련시키기](./ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)
    - 앞에서 구현한 _U-Net_ 아키텍처를 자율주행을 위한 시각 데이터 인식에 적용하고 _다이스_ 손실을 적용하고 _인스턴스 분할_을 위해 최신 알고리즘을 재사용한다. 
	
## :page_facing_up: 추가 파일

- [cityscapes_utils.py](cityscapes_utils.py): _Cityscapes_ 데이터셋을 위한 유틸리티 함수 (코드: 노트북 [6.4](./ch6_nb4_preparing_data_for_smart_car_apps.ipynb)).
- [fcn.py](fcn.py):  _FCN-8s_ 아키텍처를 함수형으로 구현 architecture (코드: 노트북 [6.5](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)).
- [keras_custom_callbacks.py](keras_custom_callbacks.py): 모델 훈련 과정을 모니터링하기 위한 맞춤형 케라스 _콜백 함수_ (코드: 노트북 [4.1](../Chapter04/ch4_nb1_implement_resnet_from_scratch.ipynb), 노트북 [6.2](./ch6_nb2_denoise_with_autoencoders.ipynb)).
- [mnist_utils.py](mnist_utils.py): `tensorflow-datasets`을 사용한, _MNIST_ 데이터셋을 위한 유틸리티 함수. (코드: 노트북 [6.1](./ch6_nb1_discover_autoencoders.ipynb)).
- [plot_utils.py](plot_utils.py): 결과를 표시하는 유틸리티 함수 (코드: 노트북 [6.2](./ch6_nb2_denoise_with_autoencoders.ipynb)).
- [tf_losses_and_metrics.py](tf_losses_and_metrics.py): CNN을 훈련시키고 평가하기 위한 맞춤형 손실 및 지표 (코드: 노트북 [6.5](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)과 노트북 [6.6](./ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)).
- [tf_math.py](tf_math.py): 다른 스크립트에서 재사용되는 맞춤형 수학 함수 (코드: 노트북 [6.5](./ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb) and [6.6](./ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)).
- [unet.py](unet.py): _U-Net_ 아키텍처의 함수형 구현 (코드: 노트북 [6.3](./ch6_nb3_improve_image_quality_with_dae.ipynb)).
