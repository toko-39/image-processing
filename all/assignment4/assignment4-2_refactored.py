import numpy as np 
import mnist 
import matplotlib.pyplot as plt
import gzip
# import sys

# ローカルMNISTデータの読み込み
train_images = mnist.parse_idx(gzip.open("all/mnist_data/train-images-idx3-ubyte.gz", "rb"))
train_labels = mnist.parse_idx(gzip.open("all/mnist_data/train-labels-idx1-ubyte.gz", "rb"))
test_images = mnist.parse_idx(gzip.open("all/mnist_data/t10k-images-idx3-ubyte.gz", "rb"))
test_labels = mnist.parse_idx(gzip.open("all/mnist_data/t10k-labels-idx1-ubyte.gz", "rb"))

def get_shuffled_index(arr):
    index_arr = np.arange(len(arr))
    np.random.shuffle(index_arr)
    return index_arr

def get_batch(random_index): 
    batch_images = train_images[random_index].reshape(len(random_index), -1)
    batch_labels = train_labels[random_index]
    return batch_images, batch_labels

def get_one_hot_label(batch_labels, output_layer_size):
    one_hot_labels = np.zeros((batch_labels.size, output_layer_size))
    one_hot_labels[np.arange(batch_labels.size), batch_labels] = 1
    return one_hot_labels

np.random.seed(777) # シードを固定

# レイヤーの次元数を定義
input_size = train_images[0].size # 入力層: 784 (28*28)
hidden_layer_size = 100 # 中間層: 100ユニット
output_layer_size = 10 # 出力層: 10ユニット (0-9の数字に対応)

# 重みとバイアスを正規分布で初期化

is_load = str(input('ロードしますか？ yes or no: '))
if is_load == 'yes' :
    loaded_data = np.load('assignment3_parameter.npz')
    weight1 = loaded_data['weight1']
    bias1 = loaded_data['bias1']
    weight2 = loaded_data['weight2']
    bias2 = loaded_data['bias2']
else:
    # 第1層（入力層 -> 中間層）
    weight1 = np.random.normal(loc=0.0, scale=np.sqrt(1 / input_size), size=(hidden_layer_size, input_size))
    bias1 = np.random.normal(loc=0.0, scale=np.sqrt(1 / input_size), size=hidden_layer_size)

    # 第2層（中間層 -> 出力層）
    weight2 = np.random.normal(loc=0.0, scale=np.sqrt(1 / hidden_layer_size), size=(output_layer_size, hidden_layer_size))
    bias2 = np.random.normal(loc=0.0, scale=np.sqrt(1 / hidden_layer_size), size=output_layer_size)

# --- 活性化関数と出力関数 ---

def ReLU(arr):
    """課題4-1 ReLU活性化関数"""
    new_arr = np.where(arr > 0, arr, 0)
    return new_arr

def softmax(x):
    """
    ソフトマックス関数（オーバーフロー対策版）
    """
    alpha = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - alpha)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# --- 順伝播の実行(重みを更新) ---
def forward_propagation(input_vector, weight1, bias1, weight2, bias2):
    # デフォルト（ドロップアウトなし）
    hidden_layer_input = np.dot(input_vector, weight1.T) + bias1
    hidden_layer_output = ReLU(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weight2.T) + bias2
    final_output = softmax(output_layer_input)

    return final_output, hidden_layer_input, hidden_layer_output

def forward_propagation_train(input_vector, weight1, bias1, weight2, bias2, ignore_number):

    hidden_layer_input = np.dot(input_vector, weight1.T) + bias1
    hidden_layer_output = ReLU(hidden_layer_input)

    # ドロップアウト率の計算 (p = ドロップアウトする割合)
    p = len(ignore_number) / hidden_layer_size
    keep_prob = 1.0 - p

    #  ドロップアウト処理（出力を0にする）
    for index in ignore_number:
        hidden_layer_output[:, index] = 0
    
    # スケーリング
    if keep_prob > 0 and keep_prob < 1.0:
        hidden_layer_output /= keep_prob

    output_layer_input = np.dot(hidden_layer_output, weight2.T) + bias2
    final_output = softmax(output_layer_input)
    return final_output, hidden_layer_input, hidden_layer_output 

def forward_propagation_test(input_vector, weight1, bias1, weight2, bias2, ignore_number):

    hidden_layer_input = np.dot(input_vector, weight1.T) + bias1
    hidden_layer_output = ReLU(hidden_layer_input)
    hidden_layer_output *= 1 - (len(ignore_number) / hidden_layer_size)
    output_layer_input = np.dot(hidden_layer_output, weight2.T) + bias2
    final_output = softmax(output_layer_input)
    return final_output, hidden_layer_input, hidden_layer_output 

def get_predicted_class(output_probabilities):
# 出力された確率から最も高い確率を持つクラス（予測結果）を取得
    if output_probabilities.ndim == 1:
        return np.argmax(output_probabilities)
    else:
        return np.argmax(output_probabilities, axis=1)

def get_cross_entropy_error(y_pred, y_true):
    
    delta = 1e-7
    loss = -np.sum(y_true * np.log(y_pred + delta))
    batch_size = y_pred.shape[0]
    cross_entropy_error = loss / batch_size

    return cross_entropy_error

def backward_propagation_and_update(batch_image_vector, hidden_layer_input, hidden_layer_output, output_probabilities, one_hot_labels, 
                                    weight1, bias1, weight2, bias2, learning_rate):
    # ドロップアウトなしの逆伝播（使われていないが、既存関数の維持）
    current_batch_size = batch_image_vector.shape[0]
    
    dEn_dak = (output_probabilities - one_hot_labels) / current_batch_size
    dEn_dX = np.dot(dEn_dak, weight2)
    dEn_dW_1 = np.dot(dEn_dak.T, hidden_layer_output)
    dEn_db_1 = np.sum(dEn_dak, axis = 0)
    
    differentiated_input = np.where(hidden_layer_input > 0, 1, 0)
    dEn_dX_sig = dEn_dX * differentiated_input
    
    dEn_dW_2 = np.dot(dEn_dX_sig.T, batch_image_vector)
    dEn_db_2 = np.sum(dEn_dX_sig, axis=0)

    weight1 -= dEn_dW_2 * learning_rate
    bias1 -= dEn_db_2 * learning_rate
    weight2 -= dEn_dW_1 * learning_rate
    bias2 -= dEn_db_1 * learning_rate
    
    return weight1, bias1, weight2, bias2

def backward_propagation_and_update_train(batch_image_vector, hidden_layer_input, hidden_layer_output,
                                        output_probabilities, one_hot_labels,
                                        weight1, bias1, weight2, bias2, learning_rate, ignore_number):
    current_batch_size = batch_image_vector.shape[0]
    dEn_dak = (output_probabilities - one_hot_labels) / current_batch_size
    dEn_dX = np.dot(dEn_dak, weight2)
    dEn_dW_1 = np.dot(dEn_dak.T, hidden_layer_output)
    dEn_db_1 = np.sum(dEn_dak, axis=0)
    
    differentiated_input = np.where(hidden_layer_input > 0, 1, 0)
    
    for index in ignore_number:
        dEn_dX[:, index] = 0
        differentiated_input[:, index] = 0
    
    dEn_dX_sig = dEn_dX * differentiated_input
    dEn_dW_2 = np.dot(dEn_dX_sig.T, batch_image_vector)
    dEn_db_2 = np.sum(dEn_dX_sig, axis=0)
    
    weight1 -= dEn_dW_2 * learning_rate
    bias1 -= dEn_db_2 * learning_rate
    weight2 -= dEn_dW_1 * learning_rate
    bias2 -= dEn_db_1 * learning_rate
    return weight1, bias1, weight2, bias2

def get_accuracy(y_prop, y_true): # 正答率計算

    y_pred = get_predicted_class(y_prop)
    accuracy = np.sum(y_pred == y_true) / len(y_prop)

    return accuracy

def calculate_accuracy_for_epoch(images, labels, weight1, bias1, weight2, bias2, mode, ignore_number):
    """
    指定されたデータセットに対するモデルの正答率を計算する関数。
    精度計算時は必ず 'test' モード（スケーリングなしの順伝播）を使用する。
    """
    images_vector = images.reshape(len(images), -1)

    if mode == 'train':

        probabilities, _, _ = forward_propagation_train(images_vector, weight1, bias1, weight2, bias2, ignore_number)
    elif mode == 'test':
        # テストデータに対する精度の計算時は、スケーリングを適用したforward_propagation_testを使用
        probabilities, _, _ = forward_propagation_test(images_vector, weight1, bias1, weight2, bias2, ignore_number)
    else:
         # デフォルト
        probabilities, _, _ = forward_propagation(images_vector, weight1, bias1, weight2, bias2)

    accuracy = get_accuracy(probabilities, labels)

    return accuracy
# --- メイン処理 ---
if __name__ == "__main__":

    batch_size = 100
    epoch_number = 10
    learning_rate = 0.01
    train_loss_list, train_acc_list, test_acc_list = [], [], []

    # ⚠️ 入力チェックをexit()からループに変更し、クラッシュを防ぐ
    while True:
        mode = str(input('実行モードを入力してください (train or test): '))
        if mode in ['train', 'test']:
            break
        print("無効なモードです。'train' または 'test' を入力してください。")

    while True:
        try:
            ignore_number = int(input(f'Dropoutの個数を 0 ~ {hidden_layer_size} で入力してください: '))
            if 0 <= ignore_number <= hidden_layer_size:
                break
            else:
                print(f"無効なドロップアウト数です。0から{hidden_layer_size}の範囲で入力してください。")
        except ValueError:
            print("無効な入力です。整数を入力してください。")

# 訓練モードの場合にのみ学習を実行
if mode == 'train':
    print("\n--- 訓練モード実行中 ---")

    for i in range(1, epoch_number + 1):
        error_sum = 0
        train_accuracy_sum = 0
        shuffled_train_image_index = get_shuffled_index(train_images)
        
        for j in range(0, len(shuffled_train_image_index), batch_size):

            # hidden_layer_size分のインデックス配列からignore_number個ランダムに選択
            random_selection = np.random.choice(np.arange(hidden_layer_size), size=ignore_number, replace=False)
            
            index = shuffled_train_image_index[j:j + batch_size]

            batch_image_vector, batch_labels = get_batch(index)

            # 順伝播を実行（Inverted Dropoutのスケーリングが適用される）
            output_probabilities, hidden_layer_input, hidden_layer_output = forward_propagation_train(
                batch_image_vector, weight1, bias1, weight2, bias2, random_selection
            )
            one_hot_labels = get_one_hot_label(batch_labels, output_layer_size)

            calculated_error = get_cross_entropy_error(output_probabilities, one_hot_labels)
            error_sum += calculated_error
            
            # 逆伝播
            weight1, bias1, weight2, bias2 = backward_propagation_and_update_train(
                batch_image_vector, hidden_layer_input, hidden_layer_output, output_probabilities, one_hot_labels,
                weight1, bias1, weight2, bias2, learning_rate, random_selection
            )
            train_accuracy_sum = calculate_accuracy_for_epoch(batch_image_vector, batch_labels, weight1, bias1, weight2, bias2, 'train', random_selection)
        
        # エポック終了後の精度計算（forward_propagation_testを使用し、Dropoutなし）
        ignore_index_for_acc = np.arange(hidden_layer_size)[:ignore_number]
        test_accuracy = calculate_accuracy_for_epoch(test_images, test_labels, weight1, bias1, weight2, bias2, 'test', ignore_index_for_acc)
    
        num_batches = len(train_images) // batch_size
        train_loss_list.append(error_sum / num_batches)
        train_acc_list.append(train_accuracy_sum / num_batches)
        test_acc_list.append(test_accuracy)
        print(f"{i}エポック目")
        print(f"  平均クロスエントロピー誤差: {error_sum / num_batches}")
        print(f"  学習データに対する正答率: {train_accuracy_sum / num_batches}")
        print(f"  テストデータに対する正答率: {test_accuracy}")
    
    # --- グラフの描画 ---
    x = np.arange(1, epoch_number + 1)
    plt.figure(figsize=(12, 5))

    # 誤差のグラフ
    plt.subplot(1, 2, 1)
    plt.plot(x, train_loss_list, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # 正答率のグラフ
    plt.subplot(1, 2, 2)
    plt.plot(x, train_acc_list, marker='o', label='Train Accuracy')
    plt.plot(x, test_acc_list, marker='s', label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    np.savez('assignment3_parameter.npz', weight1=weight1, bias1=bias1, weight2=weight2, bias2=bias2)
    
# テストモードの場合にのみ予測を実行
elif mode == 'test':
    print("\n--- テストモード実行中 ---")
    # ignore_number はテスト時のスケーリング計算に必要。ここではランダムに選ぶ必要はないが、引数として渡す
    random_selection = np.arange(hidden_layer_size)[:ignore_number]
    # テストデータに対する最終的な正答率を計算（forward_propagation_testを使用）
    test_accuracy = calculate_accuracy_for_epoch(test_images, test_labels, weight1, bias1, weight2, bias2, 'test', random_selection)

    print(f"\nテストデータに対する最終正答率: {test_accuracy}")
    print(f"（ドロップアウト数 {ignore_number} 個）")