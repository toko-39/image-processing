import numpy as np 
import mnist 
# import sys

# データの読み込み
test_images = mnist.download_and_parse_mnist_file("/mnt/c/Users/Owner/Downloads/t10k-images-idx3-ubyte.gz") 
test_labels = mnist.download_and_parse_mnist_file("/mnt/c/Users/Owner/Downloads/t10k-labels-idx1-ubyte.gz")

# # --- ユーザー入力と前処理 ---
# def get_input_image_vector():
#     # ユーザーから画像番号を入力させ、対応する画像ベクトルを返す
#     try:
#         image_number = int(input("0~9999までの整数を入力してください："))
#         if not (0 <= image_number <= 9999):
#             print("無効な数値です。")
#             sys.exit()
        
#         # 28x28の画像を1次元ベクトルに変換
#         return test_images[image_number].reshape(-1)
#     except ValueError:
#         print("無効な入力です。整数を入力してください。")
#         sys.exit()

def get_random_index(batch_size): #インデックスをランダムに取得
    test_images_arrays = np.arange(len(test_images))
    random_index = np.random.choice(test_images_arrays, size=batch_size, replace=False)
    return random_index

def get_batch_image_vector(random_index, batch_image_number):  #ベクトルを取得
    batch_images = test_images[random_index]
    return batch_images.reshape(batch_image_number, -1)

def get_batch_image_label(random_index, batch_image_number): #ラベルを取得
    batch_labels = test_labels[random_index]
    return batch_labels

def get_one_hot_label(batch_labels, output_layer_size):
    one_hot_labels = np.zeros((batch_labels.size, output_layer_size)) # ゼロで満たされた配列を作成
    one_hot_labels[np.arange(batch_labels.size), batch_labels] = 1 # 各行の、正解ラベルに対応するインデックスを1にする
    return one_hot_labels


np.random.seed(777) # シードを固定

# レイヤーの次元数を定義
input_size = test_images[0].size  # 784 (28*28)
hidden_layer_size = 100
output_layer_size = 10

# 重みとバイアスを正規分布で初期化
# 第1層（入力層 -> 中間層）
weight1 = np.random.normal(loc=0.0, scale=np.sqrt(1 / input_size), size=(hidden_layer_size, input_size))
bias1 = np.random.normal(loc=0.0, scale=np.sqrt(1 / input_size), size=hidden_layer_size)

# 第2層（中間層 -> 出力層）
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
    
    # 中間層の計算: 活性化関数の入力
    hidden_layer_input = np.dot(input_vector, weight1.T) + bias1
    
    # 中間層の出力: 活性化関数を適用
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # 出力層の計算: 活性化関数の入力
    output_layer_input = np.dot(hidden_layer_output, weight2.T) + bias2
    
    # 出力層の出力: ソフトマックスを適用して確率を算出
    final_output = softmax(output_layer_input)
    
    return final_output

# def get_predicted_class(output_probabilities):
#     # 出力された確率から最も高い確率を持つクラス（予測結果）を取得
#     return np.argmax(output_probabilities)

def get_cross_entropy_error(y_pred, y_true):
    
    delta = 1e-7
    
    loss = -np.sum(y_true * np.log(y_pred + delta)) # logの中身が0にならないようにdeltaを導入
    
    # ミニバッチサイズBで割って平均を求める
    batch_size = y_pred.shape[0]
    
    cross_entropy_error = loss / batch_size
    
    return cross_entropy_error

# --- メイン処理 ---
if __name__ == "__main__":

    batch_size = 100
    
    #インデックスをランダムに取得
    random_index = get_random_index(batch_size)
    
    #100枚のミニバッチを取り出す
    batch_image_vector = get_batch_image_vector(random_index, batch_size)
    
    #対応したラベルを取り出す
    batch_labels = get_batch_image_label(random_index, batch_size)
    
    # 順伝播を実行
    output_probabilities = forward_propagation(batch_image_vector)
    
    # one-hot labelsを取得
    one_hot_labels = get_one_hot_label(batch_labels, output_layer_size)
    
    # クロスエントロピー誤差平均を計算
    calculated_error = get_cross_entropy_error(output_probabilities, one_hot_labels)
    
    print(f"予測されたクロスエントロピー誤差は: {calculated_error} です。")
    
