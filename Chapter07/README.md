**> Chapter 7:**

# 복합적이고 불충분한 데이터셋에서 훈련시키기

새로운 인식 모델을 개발하기 위한 첫 번째 작업은 훈련 데이터셋을 수집하고 준비하는 것이다. 예전에는 무거운 훈련 단계 동안 적절하게 데이터가 흐르게 하는 파이프라인을 구성하는 것이 예술의 경지에 있는 일이었지만, 7장 첫 번째 노트북에서 보여주듯이 최근 텐서플로 기능은 이러한 복합적인 데이터를 가져와 전처리하는 일을 매우 단순화했다. 그렇지만 종종 훈련 데이터를 사용하지 못할 수 있다. 나머지 노트북은 이러한 시나리오를 다루고 다양한 해법을 제시한다. 

## :notebook: 노트북

(팁: 노트북을 시각화할 때 `nbviewer`를 사용하는 것이 좋다: `nbviewer.jupyter.org`에서 계속하려면 [여기](https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-Tensorflow/blob/master/ch7)를 클릭하라.)

- 7.1 - [`tf.data`로 효율적인 입력 파이프라인 구성하기](./ch7_nb1_set_up_efficient_input_pipelines_with_tf_data.ipynb)
    - `tf.data` API의 최신 기능을 활용해 최적화된 입력 파이프라인을 설정해 모델을 학습시킨다. 
- 7.2 - [TFRecord 생성 및 파싱](./ch7_nb2_generate_and_parse_tfrecords.ipynb)
    - 전체 데이터셋을 TFRecords로 전환하고 이 파일은 효율적으로 파싱하는 방법을 알아본다.  
- 7.3 - [3D 모델의 이미지를 렌더링하기](./ch7_nb3_render_images_from_3d_models.ipynb)
    - 3D 데이터로부터 다양한 이미지를 생성하기 위해, _OpenGL_ 기반의 `vispy`를 사용해 3D 렌더링하는 방법을 간단히 훑어본다. 
- 7.4 - [합성 이미지에서 분할 모델 훈련시키기](./ch7_nb4_train_segmentation_model_on_synthetic_images.ipynb)
    - 사전에 렌더링된 데이터셋을 사용해 모델을 훈련하고, 모델 최종 정확도에 대한 *현실성과의 격차*의 효과를 검증한다. 
- 7.5 - [단순한 도메인 적대 신경망(Domain Adversarial Network) 훈련시키기](./ch7_nb5_train_a_simple_domain_adversarial_network_(dann).ipynb)
    - 유명한 도메인 적응 기법인 *DANN*을 살펴보고 구현한다.  
- 7.6 - [DANN을 적용해 합성 데이터에 분할 모델을 훈련하기](./ch7_nb6_apply_dann_to_train_segmentation_model_on_synthetic_data.ipynb)
    - *현실성과의 격차* 문제가 있는 분할 모델의 성능을 개선하기 위해 앞에서 구현한 DANN을 적용한다.  
- 7.7 - [VAE로 이미지 생성하기](./ch7_nb7_generate_images_with_vae_models.ipynb)
    - 첫 번째 생성 신경망으로, 숫자 이미지를 생성하는 간단한 변분 오토인코더(Variational Auto-Encoder, VAE)를 만든다.  
- 7.8 - [GAN으로 이미지 생성하기](./ch7_nb8_generate_images_with_gan_models.ipynb)
    - 생성 신경망으로 유명한 생성적 적대 신경망(Generative Adversarial Networks, GAN)을 살펴본다.  
	
## :page_facing_up: 추가 파일

- [cityscapes_utils.py](cityscapes_utils.py): _Cityscapes_ 데이터셋을 위한 유틸리티 함수 (코드: 노트북 [6.4](../Chapter06/ch6_nb4_preparing_data_for_smart_car_apps.ipynb)).
- [fcn.py](fcn.py): _FCN-8s_ 아키텍처를 함수형으로 구현 (코드: 노트북 [6.5](../Chapter06/ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)).
- [keras_custom_callbacks.py](keras_custom_callbacks.py): 모델 훈련 과정을 모니터링하기 위한 맞춤형 케라스 _콜백 함수_ (코드: 노트북 [4.1](../Chapter04/ch4_nb1_implement_resnet_from_scratch.ipynb) and [6.2](./ch6_nb2_denoise_with_autoencoders.ipynb)).
- [plot_utils.py](plot_utils.py): 결과를 표시하는 유틸리티 함수 (code presented in notebook [6.2](../Chapter06/ch6_nb2_denoise_with_autoencoders.ipynb)).
- [renderer.py](renderer.py): 3D 모델로부터 이미지를 렌더링하는 객체 지향형 파이프라인 (코드: 노트북 [7.3](./ch7_nb3_render_images_from_3d_models.ipynb)).
- [synthia_utils.py](synthia_utils.py): _SYNTHIA_ 데이터셋을 위한 유틸리티 함수 (코드: 노트북 [7.4](./ch7_nb4_train_segmentation_model_on_synthetic_images.ipynb)).
- [tf_losses_and_metrics.py](tf_losses_and_metrics.py): CNN을 훈련/검증하기 위한 맞춤형 손실과 지표 (코드: 노트북 [6.5](../Chapter06/ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)과 노트북 [6.6](../Chapter06/ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)).
- [tf_math.py](tf_math.py): 다른 스크립트에서 재사용되는 맞춤형 수학 함수 (코드: 노트북 [6.5](../Chapter06/ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)과 노트북 [6.6](../Chapter06/ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)).
