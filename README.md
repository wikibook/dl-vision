# 실전! 텐서플로2를 활용한 딥러닝 컴퓨터 비전

**텐서플로 2.0과 케라스로 딥러닝을 활용해 강력한 이미지 처리 앱을 생성한다.**

《실전! 텐서플로2를 활용한 딥러닝 컴퓨터 비전》은 객체 탐지, 분할, 동영상 처리, 스마트폰 애플리케이션 등을 위한 고성능 시스템을 만드는 실습서다. 이 책은 구글이 내놓은 오픈소스 머신러닝 라이브러리의 새로운 버전인 텐서플로 2를 기반으로 한다. 

이 저장소는 각 장의 내용을 설명하는 몇 가지 노트북과 함께 책에서 보여주는 프로젝트에 대한 전체 소스 코드를 제공한다. *이 저장소는 책을 보완하기 위한 것이다. 따라서 더 자세한 설명과 고급 팁*에 대해서는 책 내용을 확인하는 것이 좋다.

## :mag_right: 이 책이 다루는 내용

컴퓨터 비전 솔루션은 점차 보편화되고 있어 의료, 자동차, 소셜미디어, 로봇 등의 분야에서 길을 개척하고 있다. 이 책은 머신러닝을 위한 구글 오픈소스 프레임워크의 새로운 버전인 텐서플로2를 살펴본다. 시각적 작업을 위해 합성곱 신경망(Convolutional Neural Network, CNN)을 활용하는 방법을 이해하게 될 것이다. 

《실전! 텐서플로2를 활용한 딥러닝 컴퓨터 비전》은 컴퓨터 비전과 딥러닝 기초로 시작해 신경망을 처음부터 만드는 방법을 알려준다. 직관적인 케라스 인터페이스와 함께 텐서플로가 가장 널리 사용되는 AI 라이브러리로 자리잡게 한 기능들을 살펴보고, 계속해서 CNN을 효율적으로 만들고, 훈련시키고, 배포하는 방법을 알아본다. 또한 구체적인 코드 예제로 이 책은 Inception과 ResNet 같은 최신 솔루션으로 이미지를 분류하고, 욜로(You Only Look Once, YOLO), Mask R-CNN, U-Net을 사용해 특정 콘텐츠를 추출하는 방법을 설명한다. 또한 생성적 적대 신경망(Generative Adversarial Networks, GANs), 변분 오토인코더(Variational Auto-Encoders, VAEs)를 만들어 이미지를 생성하고 편집하고, LSTM을 만들어 동영상을 분석한다. 이 과정에서 전이 학습, 데이터 보강, 도메인 적응, 모바일/웹 배포 등의 핵심 개념에 대한 높은 식견을 얻게 될 것이다. 이 책을 마치면, 텐서플로 2.0으로 컴퓨터 비전 문제를 해결하기 위한 이론적 이해와 실용적 기술을 모두 얻게 될 것이다.

이 책은 다음의 흥미로운 내용을 다룬다.
* 처음부터 신경망을 생성한다.
* Inception, ResNet 등 최신 아키텍처로 이미지를 분류한다. 
* YOLO, Mask R-CNN, U-Net으로 이미지 안의 객체를 탐지하고 분할한다. 
* 자율 주행 자동차와 표정 인식 시스템에서 마주하게 되는 문제를 해결한다. 
* 전이 학습, GAN, 도메인 적응으로 애플리케이션 성능을 향상시킨다. 
* 순환 신경망을 사용해 동영상 분석을 수행한다. 
* 모바일 기기와 브라우저에서 신경망을 최적화하고 배포한다. 

## :wrench: 대상 독자 및 환경

딥 러닝을 처음 접하고 이미지 파일 읽기/쓰기와 픽셀 편집과 같은 파이썬 프로그래밍과 이미지 처리에 대한 배경 지식이 있다면, 이 책이 적합하다. 새로운 TensorFlow 2 기능에 대해 궁금한 전문가라도 이 책이 도움이 될 것이다. 일부 이론적 설명에는 대수 및 미적분 지식이 필요하지만 이 책은 자율 주행 자동차의 시각 인식 및 스마트 폰 앱과 같은 실용적인 응용 프로그램에 중점을 둔 학습자를 위한 구체적인 예를 다룬다.

코드는 [주피터](http://jupyter.org/) 노트북 형태로 제공한다. 따로 표기되지 않은 경우, 파이썬 3.5(이상) 및 텐서플로 2.0을 사용해 실행하면 된다. 설치 방법은 이 책에서 설명한다. ([numpy](http://www.numpy.org/), [matplotlib](https://matplotlib.org) 등과 같이 의존성을 관리하기 위해 [Anaconda](https://anaconda.org/)를 사용하는 것이 좋다.)

아래에서 설명하듯이, 여기에서 제공하는 주피터 노트북은 직접 연구하거나 책에서 제시된 실험을 실행하고 재현하기 위한 코드 레시피로 사용될 수 있다.

또한 이 책에서 사용된 화면/도표의 컬러 이미지를 담은 PDF 파일도 제공한다. [여기를 클릭해 파일을 내려 받으면 된다.](https://www.packtpub.com/sites/default/files/downloads/9781788830645_ColorImages.pdf).

### 온라인에서 주피터 노트북 학습하기

제공된 코드와 결과를 살펴보고 싶다면 온라인으로 책의 깃허브 저장소를 접속하면 된다. 실제로 깃허브는 주피터 노트북을 렌더링하고 정적 웹 페이지로 표시 할 수 있다.
그렇지만 깃허브 뷰어는 일부 스타일 서식과 대화형 콘텐츠를 무시한다. 이를 해결하고 최상의 상태로 보려면, 온라인에 업로드된 주피터 노트북을 읽기 위해 사용할 수 있는 공식 웹 플랫폼인 [주피터 nbviewer](https://nbviewer.jupyter.org)를 사용하는 것이 좋다. 이 웹사이트를 사용해 깃허브 저장소에 저장된 노트북을 렌더링할 수 있다. 따라서 여기서 제공하는 주피터 노트북은 다음 주소에서 읽을 수 있다.
https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2

### 컴퓨터에서 주피터 노트북 실행하기

컴퓨터에서 이 문서를 읽고 실행하려면 먼저 주피터 노트북을 설치해야 한다. 파이썬 환경을 관리하고 배포하기 위해 이미 [아나콘다](https://www.anaconda.com)를 사용하고 있다면(이 책에서 권장하는 바이다) 주피터 노트북을 바로 사용할 수 있다(아나콘다와 함께 설치되기 때문이다). 다른 파이썬 배포판을 사용하고 주피터 노트북이 익숙하지 않다면, [설치 방법과 튜토리얼](https://jupyter.org/documentation)을 읽어보기 바란다.

주피터 노트북을 컴퓨터에 설치했다면 이 책의 코드 파일이 포함된 디렉터리를 찾은 다음, 터미널을 열고 다음 명령어를 실행하라. 

    $ jupyter notebook
    
웹 인터페이스는 각자 설정한 기본 브라우저에서 열린다. 거기에서 이 디렉터리를 찾아가 제공된 주피터 노트북을 열어서 읽고, 실행하거나 편집할 수 있다. 

어떤 문서는 고성능 컴퓨팅 파워를 요구할 수 있는 고급 실험(규모가 큰 데이터셋에서 인식 알고리즘을 훈련하는 등)을 포함한다. 적절하게 하드웨어를 가속화하지 않고는(즉, 2장, _텐서플로 기초와 모델 훈련_에서 설명하듯이 적절한 NVIDIA GPU가 없다면), 이 스크립트를 수행하는 데 수 시간 또는 수 일이 걸릴 수 있다(적절한 GPU가 있더라도 가장 고급 예제는 꽤 많은 시간이 걸릴 수 있다). 

### 구글 코랩에서 주피터 노트북 실행하기

주피터 노트북을 직접 실행하거나 새로운 실험을 하고 싶지만 컴퓨터가 충분한 컴퓨팅 파워를 갖추지 못했다면 [Google Colab](https://colab.research.google.com)을 사용하는 것이 좋다. 코랩은 강력한 컴퓨터에서 계산 집약적인 스크립트를 실행하고자 하는 사람을 위해 구글에서 제공하는 클라우드 기반의 주피터 환경이다. 

### 소프트웨어, 하드웨어 목록 

다음 소프트웨어와 하드웨어를 사용하면 이 책(1~9장)에서 제시하는 코드 파일을 모두 실행할 수 있다. 

| 장     | 소프트웨어 요구사항                                 | OS 요구사항                       |
| ------ | --------------------------------------------------- | --------------------------------- |
| 1~9    | Jupyter Notebook                                    | 윈도우, 맥OS X, 리눅스 중 한 가지 |
| 1~9    | Python 3.5 이상, NumPy, Matplotlib, Anaconda (선택) | 윈도우, 맥OS X, 리눅스 중 한 가지 |
| 2~9    | TensorFlow, tensorflow-gpu                          | 윈도우, 맥OS X, 리눅스 중 한 가지 |
| 3      | Scikit-Image                                        | 윈도우, 맥OS X, 리눅스 중 한 가지 |
| 4      | TensorFlow Hub                                      | 윈도우, 맥OS X, 리눅스 중 한 가지 |
| 6      | pydensecrf library                                  | 윈도우, 맥OS X, 리눅스 중 한 가지 |
| 7      | Vispy, Plyfile                                      | 윈도우, 맥OS X, 리눅스 중 한 가지 |
| 8      | opencv-python, tqdm, scikit-learn                   | 윈도우, 맥OS X, 리눅스 중 한 가지 |
| 9      | Android Studio, Cocoa Pods, Yarn                    | 윈도우, 맥OS X, 리눅스 중 한 가지 |

## :books: 목차

- 1장 - [컴퓨터 비전과 신경망](/Chapter01)
    - 1.1 - [신경망을 처음부터 만들고 훈련시키기](./Chapter01/ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)
- 2장 - [텐서플로 기초와 모델 훈련](/Chapter02)
    - 2.1 - [케라스로 모델 훈련시키기](./Chapter02/ch2_nb1_mnist_keras.ipynb)
- 3장 - [현대 신경망](/Chapter03)
    - 3.1 - [CNN 기초 연산 살펴보기](./Chapter03/ch3_nb1_discover_cnns_basic_ops.ipynb)
    - 3.2 - [텐서플로 2와 케라스로 첫 CNN을 만들고 훈련시키기](./Chapter03/ch3_nb2_build_and_train_first_cnn_with_tf2.ipynb)
    - 3.3 - [고급 최적화 모델 실험](./Chapter03/ch3_nb3_experiment_with_optimizers.ipynb)
    - 3.4 - [CNN에 정규화 기법 적용](./Chapter03/ch3_nb4_apply_regularization_methods_to_cnns.ipynb)
- 4장 - [유력한 분류 모델](/Chapter04)
    - 4.1 - [ResNet을 처음부터 구현하기](./Chapter04/ch4_nb1_implement_resnet_from_scratch.ipynb)
    - 4.2 - [케라스 애플리케이션의 모델 재사용하기](./Chapter04/ch4_nb2_reuse_models_from_keras_apps.ipynb)
    - 4.3 - [텐서플로 허브에서 모델 가져오기](./Chapter04/ch4_nb3_fetch_models_from_tf_hub.ipynb)
    - 4.4 - [전이 학습 적용하기](./Chapter04/ch4_nb4_apply_transfer_learning.ipynb)
    - 4.5 - (Appendix) [ImageNet과 Tiny-ImageNet 살펴보기](./Chapter04/ch4_nb5_explore_imagenet_and_its_tiny_version.ipynb)
- 5장 - [객체 탐지 모델](/Chapter05)
    - 5.1 - [YOLO 추론 실행하기](./Chapter05/ch5_nb1_yolo_inference.ipynb)
    - 5.2 - YOLO 모델 훈련 (미완성)
- 6장 - [이미지 보강 및 분할](./Chapter06)
    - 6.1 - [오토인코더 알아보기](./Chapter06/ch6_nb1_discover_autoencoders.ipynb)
    - 6.2 - [오토인코더로 노이즈 제거하기](./Chapter06/ch6_nb2_denoise_with_autoencoders.ipynb)
    - 6.3 - [심층 오토인코더로 이미지 품질 개선하기(초해상도)](./Chapter06/ch6_nb3_improve_image_quality_with_dae.ipynb)
    - 6.4 - [자율주행 자동차 애플리케이션을 위한 데이터 준비하기](./Chapter06/ch6_nb4_preparing_data_for_smart_car_apps.ipynb)
    - 6.5 - [의미론적 분할을 위한 FCN-8s 모델을 만들고 훈련시키기](./Chapter06/ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb)
    - 6.6 - [객체와 인스턴스 분할을 위한 U-Net 모델을 만들고 훈련시키기](./Chapter06/ch6_nb6_build_and_train_a_unet_for_urban_object_and_instance_segmentation.ipynb)
    - 6.6 - [Object and Instance Segmentation for Smart Cars with U-Net](./Chapter06/ch6_nb6_object_and_instance_segmentation_for_smart_cars_with_unet.ipynb)
- 7장 - [복합적이고 불충분한 데이터셋에서 훈련시키기](/Chapter07)
    - 7.1 - [`tf.data`로 효율적인 입력 파이프라인 구성하기](./Chapter07/ch7_nb1_set_up_efficient_input_pipelines_with_tf_data.ipynb)
    - 7.2 - [TFRecord 생성 및 파싱](./Chapter07/ch7_nb2_generate_and_parse_tfrecords.ipynb)
    - 7.3 - [3D 모델의 이미지를 렌더링하기](./Chapter07/ch7_nb3_render_images_from_3d_models.ipynb)
    - 7.4 - [합성 이미지에서 분할 모델 훈련시키기](./Chapter07/ch7_nb4_train_segmentation_model_on_synthetic_images.ipynb)
    - 7.5 - [단순한 도메인 적대 신경망(Domain Adversarial Network) 훈련시키기](./Chapter07/ch7_nb5_train_a_simple_domain_adversarial_network_(dann).ipynb)
    - 7.6 - [DANN을 적용해 합성 데이터에 분할 모델을 훈련하기](./Chapter07/ch7_nb6_apply_dann_to_train_segmentation_model_on_synthetic_data.ipynb)
    - 7.7 - [VAE로 이미지 생성하기](./Chapter07/ch7_nb7_generate_images_with_vae_models.ipynb)
    - 7.8 - [GAN으로 이미지 생성하기](./Chapter07/ch7_nb8_generate_images_with_gan_models.ipynb) 	
- 8장 - [동영상과 순환 신경망](/Chapter08)
    - 8.1 - [LSTM을 사용해 행동 탐지하기](./Chapter08/ch8_nb1_action_recognition.ipynb)
- 9장 - [모델 최적화 및 모바일 기기 배포](/Chapter09)
    - 9.1 - [모델 프로파일링](./Chapter09/ch9_nb1_profiling.ipynb)
    - 9.2 - [비최댓값 억제 알고리즘 비교](./Chapter09/ch9_nb2_nms_speed_comparison.ipynb)
    - 9.3 - [감정 탐지 모델을 훈련시키고 모바일 기기를 위해 이 모델을 전환](./Chapter09/ch9_nb3_train_model.ipynb)
    - [iOS 앱](./Chapter09/coreml_ios)
    - [안드로이드 앱](./Chapter09/tf_lite_android)
    - [Tensorflow.js 앱](./Chapter09/tfjs)
