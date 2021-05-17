**> 1장:**
<a href="https://www.packtpub.com" title="Get the book!">
    <img src="../banner_images/book_cover.png" width=200 align="right">
</a>
# 컴퓨터 비전과 신경망

1장에서는 컴퓨터 비전과 머신 러닝을 소개하고 신경망이 작동하는 방식을 자세히 설명한다. 이 디렉토리에서는 간단한 신경망을 처음부터 구현하고 전통적인 분류 작업에 적용한다. 

## :notebook: 노트북

(팁: 노트북을 시각화할 때 'nbviewer'를 사용하는 것이 좋다: `nbviewer.jupyter.org`에서 계속하려면 [여기](https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-Tensorflow/blob/master/ch1)를 클릭하라.) 

- 1.1 - [신경망을 처음부터 구성하고 훈련시키기](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)
    - 간단한 신경망을, *인공 뉴런 모델링*부터 손으로 쓴 숫자를 분류하도록 훈련될 수 있는 *다중 계층으로 구성된 시스템*까지 구현한다. 
	
## :page_facing_up: 추가 파일

- [neuron.py](neuron.py): 정보를 앞으로 전달할 수 있는 *인공 뉴런* 모델(코드: 노트북 [1.1](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)).
- [fully_connected_layer.py](fully_connected_layer.py): 몇 개의 뉴런을 그룹핑한 함수형 *계층*을 매개변수를 최적화하는 메서드를 사용해 구현(코드: 노트북 [1.1](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)). 
- [simple_network.py](simple_network.py): 구현한 내용 모두 모듈식 *신경망*에 모두 감싼 클래스. 이 신경망은 다양한 작업을 위해 훈련시키고 사용할 수 있다.(코드: 노트북 [1.1](./ch1_nb1_build_and_train_neural_network_from_scratch.ipynb)).
