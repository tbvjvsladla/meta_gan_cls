#DataLoader을 만드는 파이썬 파일
#GPU에 직접 대량의 데이터를 넣는것이 불가능하기에
#자동화 + 최적화로 데이터를 넘겨주는 것임

from psutil import virtual_memory
#psutil은 파이썬을 위한 실행중인 프로세스 및 시스템 리소스
#그리고 정보 검색을 위한 크로스 플랫폼 라이브러임
import torch

# gpu_info = !nvidia-smi
# Jupyter Notebook의 경우 cmd 명령어를 코드라인에서
#실행시키고자 할 때는 앞에 !를 붙인다.
# gpu_info = '\n'.join(gpu_info)
# 위 코드는 jupyter notebook에서는 실행되었으나
# vscode 환경의 py에서는 실행이 안됨 -> 나중에 확인

#램 사용량 체크 -> psutil 라이브러리에 있음
ram_gb = virtual_memory().total / 1e9
#버추얼 메모리 출력이 byte형식이니 10^9승 -> 1e9로 나누어서
#GB형식으로 나타내려는 것
print('{:.1f} GB 램 사용 가능'.format(ram_gb))
#{순서} 문자열 .format(출력값)
#여러개면 : {순서1} {순서2} 문자열.format(출력값1,출력값2)

#pytorch-GPU 연결 확인
# device = torch.device("CUDA" if torch.cuda.is_available() else "CPU")
# print('학습을 진행하는 기기 : ', device)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = torch.device(device)
print("학습을 진행하는 기기 : ", device)
#위 2줄 코드는 동작을 안해서 스택오버플로으의 아래 코드 사용

from fastai.data.external import untar_data, URLs
#Fast AI라고 제레미 하워드가 만든 Pytorch의 상위 wrapper임
#비슷한거로 pytorch lightning이 있음
import glob
#외부 파일을 불러와 파이썬 작성시 사용하는 라이브러리
#사용자가 조건에 맞는 파일명을 리스트 형식으로 반환함

coco_path = untar_data(URLs.COCO_SAMPLE)
paths = glob.glob(str(coco_path) + "/train_sample/*.jpg")
print(coco_path)
#Fast AI 라이브러리 설치하면 예제 학습자료 COCO_sample를
#coco_path로 불러온 다음에 이 중 jpg 파일 경로만 path에 리스트로 저장
#목표는 딥러닝 학습자료를 따로 생성하고 러닝까지 하는 것


import numpy as np
import time
#numpy 라이브러리를 as 명령어로 np로 줄여쓰겠다. 

#np.random.seed(42) #랜덤 시드에 42입력 
np.random.seed(seed = int(time.time())) #시간으로 시드 입력
chosen_path = np.random.choice(paths, size=5000, replace=False)
#choice(a, size=None, replace=True, p=None)
index = np.random.permutation(5000)
#숫자가 입력시 해당 숫자까지의 무작위 배열을 만듬
#range(0~n) 리스트의 순번이 뒤죽박죽임

train_path = chosen_path[index[:3500]] #0~3500장까지
val_path = chosen_path[index[3500:]] #3501~끝까지
#훈련 데이터, 검증데이터 셋을 나눔

print(len(train_path))
print(len(val_path))

import matplotlib
import matplotlib.pyplot as plt
#파이썬 매트랩 라이브러리, plot 라이브러리 인포팅

sample = matplotlib.image.imread(train_path[int(np.random.random() * 3500)])
#train_path로 무작위로 셔플된 데이터 중 무작위 index 항목을 imread 하기위해
#np.random.random() 메서드 사용 => 이때 train path가 3500개 까지만 있으니
#3500을 곱한다.
plt.imshow(sample)
plt.axis("off")
plt.show()

import torch
from torch.utils.data import Dataset

#dataset로드하는 class 생성하기
# class myDataset(Dataset):
#     #기본 생성자
#     def __init__(self) -> None:
#         super().__init__()
#super()는 상속받은 부모 클래스를 의미
#만약 상속받은 클래스랑 부모 클래스가 모두 init가 있고
#여기서 상속받은 클래스가 super()을 안쓰고 init를 하면
#부모 클래스의 init에 자식init를 덮어쓰는 Overriding이 발생함

#def test(x:int)->None:
#메서드 정의 시 ':'는 매개변수의 타입에 대한 주석
#'->'는 리턴값 타입에 대한 주석이다.

#클래스 간단공부 (https://wikidocs.net/28)
#클래스로 사칙연산 프로그램을 만든다면
#0) 사칙연산을 위한 2개의 인자 받는 메서드 호출
#1) 수행할 사칙연산 메서드 호출
#이런 형식이 된다. 이때, 0)항목을 수행하지 않고 1)로 넘어가면
#속성오류 AttributeError가 발생하기에 0)메서드를 자동호출하게 해주는
#방식이 좀 더 유용하다. 이를 Constructor(생성자) 메서드라 부르며,
#__init__ 이렇게 쓴다, 반대로 소멸자는 __del__이다.

#파이썬 메서드의 첫 번째 매개변수 이름은 관례적으로 self를 사용한다. 
#객체를 호출할 때 호출한 객체 자신이 전달되기 때문에 self라는 이름를 
#사용한 것이다. 물론 self말고 다른 이름을 사용해도 상관없다.

#     def __getitem__(self, index) -> T_co:
#         return super().__getitem__(index)
#     #__init__랑 __getitem__은 vscode에서 
#     #기본 틀을 만들어주는데 확인해보기
#     def __len__(self):
#         return self.x.shape[0]
#__getitem__과 __len__은 클래스 안에 사전 정의된
#특별 메서드(special method)이다. 


class myDataset(Dataset): #myDataset 클래스는 Dataset클래스를 상속받음
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y #생성자는 x, y매개변수를 받는다는 기본문법
    
    def __getitem__(self, index):
    #__getitem__은 리스트의 슬라이싱을 돕는 특별 메서드이다.
    #이거 안쓰면 슬라이싱 할때 객체.x.[:] 같이 x속성에 직접 접근해야한다
    #이걸 쓰면 객체.[:]로 속성 생략하고 쓰는거 가능
        return self.x[index], self.y[index]

    def __len__(self): #객체 길이 반환하는 특별 메서드
        return self.x.shape[0]
        #shape : ndarray 타입 변수의 몇행, 몇열인지 반환
        #이를 통해 x 매개변수는 ndarray타입 변수가 들어감을 알 수 있다.


x = np.random.randint(0, 100, 5) #랜덤으로 0~100숫자 중 5개 추출
y = np.random.randint(0, 100, 5)
print("ndarray x, y is", x, y)

x = torch.Tensor(x)
y = torch.Tensor(y)
#텐서(tensor)는 배열(array)이나 행렬(matrix)과 매우 유사한 
#특수한 자료구조입니다. PyTorch에서는 텐서를 사용하여 모델의 
#입력과 출력뿐만 아니라 모델의 매개변수를 부호화(encode)합니다.
print("tenser x, y is ", x, y)

dataset = myDataset(x, y)
#dataset라는 객체를 myDataset클래스를 통해 인스턴스(실체화함) 
print("dataset is ", dataset)

from torch.utils.data import DataLoader

dataLoader = DataLoader(dataset, batch_size=3, num_workers=0, pin_memory=True)
#여기 부분이 GPU로 불러온 데이터를 보내는 구문임
#batch_size는 한번에 GPU로 넣는 개수를 말함
#(여기서 x, y의 크기는 최대 5이니 batch_size는 5이상 해봣자 의미없음)
#num_workers는 일하는 GPU 개수 할당
#pin_memory는 DataLoader 수행할 시 우선순위를 True/False하겠다

x, y = next(iter(dataLoader))
print(x, y)
#이터레이터(iterator)는 값을 차례대로 꺼낼 수 있는 객체(object)입니다.
#지금까지 for 반복문을 사용할 때 range를 사용했습니다. 
#만약 100번을 반복한다면 for i in range(100):처럼 만들었습니다.
#이 for 반복문을 설명할 때 for i in range(100):은 0부터 99까지 
#연속된 숫자를 만들어낸다고 했는데, 사실은 숫자를 모두 만들어 내는 것이 아니라
#0부터 99까지 값을 차례대로 꺼낼 수 있는 이터레이터를 하나만 만들어냅니다.
#이후 반복할 때마다 이터레이터에서 숫자를 하나씩 꺼내서 반복합니다.
#만약 연속된 숫자를 미리 만들면 숫자가 적을 때는 상관없지만
#숫자가 아주 많을 때는 메모리를 많이 사용하게 되므로 성능에도 불리합니다.
#그래서 파이썬에서는 이터레이터만 생성하고 값이 필요한 시점이 되었을 때 값을
#만드는 방식을 사용합니다. 즉, 데이터 생성을 뒤로 미루는 것인데
#이런 방식을 지연 평가(lazy evaluation)라고 합니다.