## 디바이스별로 텐서플로2 실행하기

![header](./images/header.jpg)

이 프로젝트는 안드로이드, iOS 및 브라우저에서 감정 인식 앱을 만드는 과정을 안내한다. 
- 데이터를 로드하고 모델을 훈련시킨다.
- SavedModel로 내보낸다.
- 텐서플로 라이트를 사용해 안드로이드에서 실행한다.
- Core ML을 사용해 iOS에서 실행한다.
- TensorFlow.js를 사용해 브라우저에서 실행한다. 

## 데모
브라우저 버전에 대한 데모는 [여기](https://ndres.me/face-emotion/)에서 확인할 수 있다.

## 브라우저에서 실행
폴더: `tfjs`

[yarn](https://yarnpkg.com/en/)을 설치하고 실행한다.

    cd tfjs
    yarn
    yarn watch
    
## iOS에서 실행
폴더: `coreml_ios`
맥 컴퓨터에 Xcode와 [Cocoa Pods](https://cocoapods.org/)이 설치돼 있어야 한다. 

    cd coreml_ios/app
    pod install
    open Core\ ML\ Demo.xcworkspace

## 안드로이드에서 실행
폴더: `tf_lite_android`

[Android Studio](https://developer.android.com/studio)가 설치돼 있어야 한다.
