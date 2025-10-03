import numpy as np 
import mnist 
import sys

# データの読み込み
test_images = mnist.download_and_parse_mnist_file("/mnt/c/Users/Owner/Downloads/t10k-images-idx3-ubyte.gz") 
test_labels = mnist.download_and_parse_mnist_file("/mnt/c/Users/Owner/Downloads/t10k-labels-idx1-ubyte.gz")

# --- ユーザー入力と前処理 ---
def get_input_image_vector():
    # ユーザーから画像番号を入力させ、対応する画像ベクトルを返す
    try:
        image_number = int(input("0~9999までの整数を入力してください："))
        if not (0 <= image_number <= 9999):
            print("無効な数値です。")
            sys.exit()
        
        # 28x28の画像を1次元ベクトルに変換
        return test_images[image_number].reshape(-1)
    except ValueError:
        print("無効な入力です。整数を入力してください。")
        sys.exit()

np.random.seed(777) # シードを固定

# レイヤーの次元数を定義
input_size = test_images[0].size  # 784 (28*28)
hidden_layer_size = 10
output_layer_size = 10

# 重みとバイアスを正規分布で初期化
# 第1層（入力層 -> 隠れ層）
weight1 = np.random.normal(loc=0.0, scale=np.sqrt(1 / input_size), size=(hidden_layer_size, input_size))
bias1 = np.random.normal(loc=0.0, scale=np.sqrt(1 / input_size), size=hidden_layer_size)

# 第2層（隠れ層 -> 出力層）
weight2 = np.random.normal(loc=0.0, scale=np.sqrt(1 / hidden_layer_size), size=(output_layer_size, hidden_layer_size))
bias2 = np.random.normal(loc=0.0, scale=np.sqrt(1 / hidden_layer_size), size=output_layer_size)

# --- 活性化関数と出力関数 ---
def sigmoid(x):
    """シグモイド活性化関数"""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """
    ソフトマックス関数（オーバーフロー対策版）
    各要素を0から1の間の確率に変換
    """
    alpha = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - alpha)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# --- 順伝播の実行 ---
def forward_propagation(input_vector):
    
    # 隠れ層の計算: 活性化関数の入力
    hidden_layer_input = np.dot(weight1, input_vector) + bias1
    
    # 隠れ層の出力: 活性化関数を適用
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # 出力層の計算: 活性化関数の入力
    output_layer_input = np.dot(weight2, hidden_layer_output) + bias2
    
    # 出力層の出力: ソフトマックスを適用して確率を算出
    final_output = softmax(output_layer_input)
    
    return final_output

def get_predicted_class(output_probabilities):
    # 出力された確率から最も高い確率を持つクラス（予測結果）を取得
    return np.argmax(output_probabilities)

# --- メイン処理 ---
if __name__ == "__main__":
    # ユーザーから入力を受け取る
    input_vector = get_input_image_vector()
    
    # 順伝播を実行
    output_probabilities = forward_propagation(input_vector)
    
    # 予測結果を取得して出力
    predicted_class = get_predicted_class(output_probabilities)
    print(f"この画像の予測結果は: {predicted_class} です。")