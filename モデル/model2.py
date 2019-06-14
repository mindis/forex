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
* 拡張畳み込みだから,dilation_rateを使用する？

def model():
    inputs = layers.Input((num_nits,1))

    cnn = layers.Conv1D(num_nits,2,padding="causal",activation="relu")(inputs)
    cnn = layers.Conv1D(num_nits*2,2,padding="causal",activation="relu",dilation_rate=2)(cnn)
    cnn = layers.Conv1D(num_nits*4,2,padding="causal",activation="relu",dilation_rate=4)(cnn)
    cnn  = layers.Conv1D(num_nits*8,2,padding="causal",activation="relu",dilation_rate=8)(cnn)

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




#####################################################################################
* 変化が小さい場合、より小さなdilation_rateは微妙は変化をとらえることができる、逆も同じ
* 異なるdilation_rateを使用する。(並列)
*  グローバル規模で情報を取得するために、大きなカーネルでいくつかの連続した畳み込みを使用します
#####################################################################################

* 同じ受容野を持つ畳み込みカーネルと比較すると、
        「拡張畳み込みはより小さなカーネルサイズで、パーラメータを減らし、受容野を広げ、ほぼ同じ情報量を経ることができる。」
* カーネルの数が大きい程より多くの情報を抽出することができる。（畳み込みカーネル ＝ フィルターサイズ）
* 残接続を用いて 、情報や勾配損失に対処する。
* 拡張畳込みにもとずく、「GRL」と「LRL」を定義しそれらの両方を統合する。（より良い再構築のために豊富な特徴マップを維持するために統合されています。）
        GRL（グローバル残差学習） ・・・ 初期情報を利用する
        LRL（ローカル残差学習） ・・・ 情報の流れをさらに改善する

def model():

    def grl(inputs):
        bn = layers.BatchNormalization()(inputs)
        cnn = layers.Conv1D(32,9,padding="same",activation="relu",dilation_rate=1)(bn)
        cnn = layers.Conv1D(64,3,padding="same",activation="relu",dilation_rate=2)(cnn)
        cnn = layers.Conv1D(32,3,padding="same",activation="relu",dilation_rate=3)(cnn)
        cnn = layers.Conv1D(64,3,padding="same",activation="relu",dilation_rate=2)(cnn)
        cnn = layers.Conv1D(32,3,padding="same",activation="reli",dilation_rate=3)(cnn)
        cnn = layers.conv1D(64,3,padding="same",activation="relu",dilation_rate=2)(cnn)
        bn2 = layers.BatchNormalization()(cnn)
        cnn = layers.Conv1D(32,3,padding="same",activation="relu",dilation_rate=3)(bn2)
        bn3 = layers.BatchNormalization()(cnn)

        concat = layers.Concatenate()([bn,bn2,bn3])
        cnn = layers.Conv1D(64,3,padding="same",activation="relu",dilation_rate=2)(concat)
        concat = layers.Concatenate()([inputs,cnn])

        return concat

    def lrl(inputs):
        bn = layers.BatchNormalization()(inputs)
        cnn = layers.Conv1D(32,9,padding="same",activation="relu")(bn)

        cnn2 = layers.Conv1D(64,3,padding="same",activation="relu",dilation_rate=2)(cnn)

        cnn3 = layers.Conv1D(32,3,padding="same",activation="relu",dilation_rate=3)(cnn2)
        concat = layers.Concatenate()([cnn3,cnn])

        cnn4 = layers.Conv1D(64, 3, padding="same", activation="relu", dilation_rate=2)(concat)
        concat2 = layers.Conv1D(cnn2, cnn4])

        cnn5 = layers.Conv1D(32,3,padding="same",activation="relu",dilation_rate=3)(concat2)
        concat3 = layers.Concatenate()([concat,cnn5])

        cnn6 = layers.Conv1D(62,3,padding="same",activation="relu",,dilation_rate=2)(concat3)
        bn2 =  layers.BatchNormalization()(cnn6)
        concat = layers.Concatenate()([concat2,bn2])

        cnn7 = layers.Conv1D(32,3,padding="same",activation="relu",dilation_rate=3)(concat)
        bn3 = layers.BatchNormalization()(cnn7)
        concat = layers.Concatenate()([concat3,bn3])

        concat = layers.Concatenate()([concat,bn,bn2])

        return concat

    inputs = layers.Input((num_nits,1))

    grl = grl(inputs)
    lrl = lrl(inputs)

    concat = layers.Concatenate()([grl,lrl])

    cnn = layers.Conv1D(32,3,padding="same",activation="relu")(concat)

    flatten = layers.Flatten()(cnn)

    outputs = layerws.Dense(1)(flatten)

    model = tf.keras.Model(inputs,outputs)

    return model
