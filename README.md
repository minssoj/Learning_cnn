# Learning_cnn
  Minso Jeong, MOOC 머신러닝 기반의 영상처리 (CNN)
 
![cnn_main](./img/cnn_main.png)

## Schedule
|          내용         |   날짜     |   비고   |
| -------------------------------- |:---------------:|--------------------------|
|01. Linear regression model <br/> Logistic model <br/>  Perceptron model | 2020. 08. 31 | Tensor |
|02. Classification model <br/> Convolutional Neural Network (CNN) | 2020. 09. 01 | Softmax, Overfitting|
|03. Convolutional Neural Network (CNN) - VGG16| 2020. 09. 02 | DataGenerator |
|04. AutoEncoder (AE) <br/> OpenCV | 2020. 09. 03 | |
|05. YOLO <br/> Generative Adversarial Network (GAN) | 2020. 09. 04 |  |

## Setting
* anaconda 설치 (https://www.anaconda.com/products/individual)
* pycharm 설치 (https://www.jetbrains.com/ko-kr/pycharm)
* 가상환경 생성
    * Anaconda Prompt
    ```
    conda create -n 가상환경이름
    ```
* 라이브러리 설치
    * package
        * tensorflow = 2.0.0 (tensorflow-gpu = 2.0.0)
        * numpy = 1.16.4
        * seaborn = 0.10.1
        * scikit-learn
        * pandas
        * matplotlib
        * scipy
     
## Data
* day2 - highlevelModel.py
    * ThoraricSurgery.csv (https://github.com/gilbutITbook/006958/tree/master/deeplearning/dataset)
* day2 - overfittingModel.py
    * wine.csv (https://github.com/gilbutITbook/006958/tree/master/deeplearning/dataset)
* day2 - sonarModel.py
    * sonar.csv (https://github.com/gilbutITbook/006958/tree/master/deeplearning/dataset)    
* day3 - DataPreprocessing.py
    * cat_and_dog (https://www.microsoft.com/en-us/download/details.aspx?id=54765)
        * train 폴더 안에 cat, dog 각각 2000장씩        
    * cat_and_dog_small의 경우 DataPreprocessing.py로 제작   
* day4 - openCVEx.*.py
    * haarcascade_frontalface_alt (https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
* day5 - yolo_inference.py
    * coco.names (https://github.com/pjreddie/darknet/tree/master/data)
    * yolov3.tf (https://docs.openvinotoolkit.org/latest/omz_models_public_yolo_v3_tf_yolo_v3_tf.html)
    * dog_example.jpg (https://github.com/pjreddie/darknet/tree/master/data)
* day5 - superResolutionModelEx.py
    * content (https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500)
* day5 - autoColoringModelEx.py
    * colorize (https://www.cs.toronto.edu/~kriz/cifar.html)
* assignment - assignment_day5.py
    * shrine_temple (https://marshal-art.jp/laboratory/2020/03/transfer-leraning-vgg16-01/)

## References
* Main image (http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf)
* YOLO presentation
    * https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p


