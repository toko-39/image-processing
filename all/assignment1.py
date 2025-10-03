import numpy as np 
import mnist 
import matplotlib.pyplot as plt 
import sys
from pylab import cm

testImages = mnist.download_and_parse_mnist_file("/mnt/c/Users/Owner/Downloads/t10k-images-idx3-ubyte.gz") 
testLabels = mnist.download_and_parse_mnist_file("/mnt/c/Users/Owner/Downloads/t10k-labels-idx1-ubyte.gz")

def inputImageNum():
    imageNumber = int(input("0~9999までの整数を入力してください："))

    if (imageNumber < 0 or imageNumber > 9999) :
        print("無効な数値です")
        sys.exit()
        
    imageVector = testImages[imageNumber].reshape(-1) # 28×28サイズの画像を多次元ベクトルに変換 reshapeメソッドでサイズを自動で計算
    return imageVector

imageVector = inputImageNum()

np.random.seed(777) # シードを固定

N = imageVector.size # 画像サイズ
dispersionImage = np.sqrt(1 / N) # 標準偏差を計算

midClass = 10 # 中間層の数
dispersionMid = np.sqrt(1 / midClass) # 標準偏差を計算

classNumber = 10 # 出力の個数

W1 = np.random.normal(loc=0.0, scale=dispersionImage, size=(midClass,N)) # (平均、分散、個数)
b1 = np.random.normal(loc=0.0, scale=dispersionImage, size=midClass)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

midInputParameter = W1 @ imageVector + b1 
midRezult = sigmoid(midInputParameter) # 中間層の計算結果

W2 = np.random.normal(loc=0.0, scale=dispersionMid, size=(classNumber,midClass))
b2 = np.random.normal(loc=0.0, scale=dispersionMid, size=classNumber)

classInput = W2 @ midRezult + b2 # ソフトマックス関数に入力する値

def softmax(a):
    alpha = np.max(a, axis=-1, keepdims=True) # 軸指定(axis=-1)で、最後の次元に沿って最大値を計算 Gemini参照
    exp_a = np.exp(a - alpha)
    sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)
    y_i = exp_a / sum_exp_a
    
    return y_i

def result(y):
    maxIndexFlat = np.argmax(y) # 最大値のインデックスを取得
    return maxIndexFlat
    
print(result(softmax(classInput)))