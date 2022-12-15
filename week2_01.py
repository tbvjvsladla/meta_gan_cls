from psutil import virtual_memory
#psutil은 파이썬을 위한 실행중인 프로세스 및 시스템 리소스
#그리고 정보 검색을 위한 크로스 플랫폼 라이브러임
import torch

# gpu_info = !nvidia-smi
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
#Fast AI라고 제레미 하워드가 만든 Pythrch의 상위 wrapper임
import glob
#외부 파일을 불러와 파이썬 작성시 사용하는 라이브러리
#사용자가 조건에 맞는 파일명을 리스트 형식으로 반환함

coco_path = untar_data(URLs.COCO_SAMPLE)
paths = glob.glob(str(coco_path) + "/train_sample/*.jpg")
print(coco_path)
#Fast AI 라이브러리 설치하면 예제 학습자료 COCO_sample를
#coco_path로 불러온 다음에 이 중 jpg 파일 경로만 path에 리스트로 저장


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

train_path = chosen_path[index[:3500]]
val_path = chosen_path[index[:3500]]

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
class myDataset(Dataset):
    #기본 생성자
    def __init__(self) -> None:
        super().__init__()
    
    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)
    #__init__랑 __getitem__은 vscode에서 
    #기본 틀을 만들어주는데 확인해보기
    
    def __len__(self):
        return self.x.shape[0]