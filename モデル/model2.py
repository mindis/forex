*  局所的特徴を抽出する

def model():
    def fine(inputs):
        cnn = Conv1D(36, 4, padding="same")(inputs)
        cnn = layers.PReLU()(cnn)
        cnn = Conv1D(36, 4, padding="same")(cnn)
        cnn = layers.PReLU()(cnn)
        cnn = Conv1D(72, 4, padding="same")(cnn)
        cnn = layers.PReLU()(cnn)

        return cnn
    def midium(inputs):
        cnn = Conv1D(36, 3, padding="same")(inputs)
        cnn = layers.PReLU()(cnn)
        cnn = Conv1D(72, 3, padding="same")(cnn)
        cnn = layers.PReLU()(cnn)

        return cnn

    def coarse(inputs):
        cnn = Conv1D(72, 2, padding="same")(inputs)
        cnn = layers.PReLU()(cnn)
        concat = layers.Concatenate()([inputs, cnn])

        return cnn
    ​
    inputs = layers.Input((num_nits, 1))
    ​
    inputs_2 = layers.MaxPool1D()(inputs)
    inputs_3 = layers.MaxPool1D(4)(inputs)
    ​
    net1 = fine(inputs=inputs)
    net2 = midium(inputs_2)
    net3 = coarse(inputs_3)
    ​
    net2 = layers.UpSampling1D()(net2)
    net3 = layers.UpSampling1D(4)(net3)
    ​​
    concat = layers.Add()([net1, net2, net3])
    ​
    concat = layers.Flatten()(concat)
    dense = layers.Dense(128, activation="relu")(concat)
    outputs = layers.Dense(s)(concat)
    ​
    model = tf.keras.Model(inputs, outputs)
    return model

* 階層型畳み込みニューラルネットワーク（階層によって高レベルの特徴を抽出できる？）
* ジャンプ接続によって低階層によって得られた情報を高階層に供給される
def model():
    inputs = layers.Input((num_nits,1))

    cnn1 = layers.Conv1D(36,8,padding="causal",activation="relu")(inputs)

    cnn2 = layers.Conv1D(36,4,padding="causal",activation="relu")(inputs)

    cnn3 = layers.Conv1D(36,2,padding="causal",activation="relu")(inputs)

    cnn4 = layers.Conv1D(108,1,padding="causal",activation="relu")(inputs)

    add = layers.Add()([cnn1,cnn2,cnn3,cnn4])

    dense1 = layers.Dense(36, activation="relu")(add)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.3)(dense1)

    dense2 = layers.Dense(36, activation="relu")(add)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Dropout(0.3)(dense2)

    concat = layers.Concatenate()([add,dense1,dense2])

    add = layers.Flatten()(concat)
    outputs = layers.Dense(1)(add)

    model = tf.keras.Model(inputs,outputs)
    tf.keras.utils.plot_model(model)

    return model


* アテンションプーリングを備えた畳み込みニューラルネットワーク
* フィルターサイズ（受容野）の大きさが精度に関係する
* ローカルプーリングは 特徴を失わせる
* 拡張畳み込みだから,diration_rateを使用する？

def model():
    inputs = layers.Input((num_nits,1))

    cnn = layers.Conv1D(num_nits,2,padding="causal",activation="relu")(inputs)
    cnn = layers.Conv1D(num_nits*2,2,padding="causal",activation="relu",diration_rate=2)(cnn)
    cnn = layers.Conv1D(num_nits*4,2,padding="causal",activation="relu",diration_rate=4)(cnn)
    cnn  = layers.Conv1D(num_nits*8,2,padding="causal",activation="relu",diration_rate=8)(cnn)

    pool1 = layers.Conv1D(10,2,padding="same",activation="relu")(cnn)

    pool2 = layer.Conv1D(10,2,padding="causal",activation="relu")(cnn)
    pool2 = layers.Conv1D(10,2,padding="causal",activation="relu")(pool2)

    multipl = layers.Multiply()([pool1,pool2])

    cnn  = layers.Conv1D(10,2,padding="causal",activation="relu")(multipl)

    flatten = layers.Flatten()(cnn)
    outputs = layers.Dense(1)(flatten)

    model = tf.keras.Model(inputs,outputs)
    tf.keras.utils.plot_model(model)
    return model


* フィルタの長さが重要な側面
* 複数の時間スケールで時間的特徴を捕捉する為に、畳み込み層に複数の長さのフィルタを組み込む
* 関連する機能を取り込むには、5のフィルタ長が有用で十分

* 転送学習は精度がゼロの状態からトレーニングするよりも精度が優れている
* Elementwise Addition == layers.Add ?

def model():
    def block(inputs,type=1):
        cnn1 = layers.Conv1D(4,2,padding="causal",activation="relu")(inputs)
        cnn2 = layers.Conv1D(8,2,padding="causal",activation="relu")(inputs)
        cnn3 = layers.Conv1D(16,2,padding="causal",activation="relu")(inputs)
        cnn4 = layers.Conv1D(32,2,padding="causal",activation="relu")(inputs)
        cnn5 = layers.Conv1D(64,2,padding="causal",activation="relu")(inputs)

        concat =  layers.Concatenate()([cnn1,cnn2,cnn3,cnn4,cnn5])

        if type  == 1:
            concat = layers.ReLU()(concat)
        return concat

    def net(inputs, type=1):
        block1 = block(inputs)
        block1 = block(block1,2)

        if type == 1:
            inputs = layers.Conv1D(124,2,padding="causal",activation="relu")(inputs)

        add = layers.Add()([block1,inputs])

        relu = layers.ReLU()(add)

        return relu

    inputs = layers.Input((num_nits,1))
    net1 = net(inputs)
    net1 = net(net,2)

    gmp = layers.GlobalMaxPool1D()(net1)
    outputs = layers.Dense(1)(gmp)

    model = tf.keras.Model(inputs,outputs)

    return model


