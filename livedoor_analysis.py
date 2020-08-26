from hmmlearn import hmm
import pickle

from text_analysis import TextAnalysis

input_dir = "./datasets/livedoor/dokujo-tsushin/"
input_data = input_dir + "dokujo-tsushin-4778030.txt"

X = TextAnalysis.mecab_analysis(input_data)

# verbose=Trueで各回のイテレーションを確認できる．
# model = hmm.MultinomialHMM(n_components=10, n_iter=1000, verbose=True)
model = hmm.MultinomialHMM(n_components=10, n_iter=1000)

model.fit(X)

L,Z = model.decode(X)
# print(model.transmat_) # 遷移確率の出力 
# print(model.monitor_) # historyの配列は最後から2つの対数尤度を出力している．
sample = model.sample(n_samples=100)

# 辞書の読み込み
with open('./datasets/livedoor/livedoor_dict.pkl', 'rb') as f:
    livedoor_dict = pickle.load(f)

# モデルからサンプルしてテキスト生成
sample_id = sample[0].flatten()
sample_text = ""
for id in sample_id:
    for key in livedoor_dict:
        if id == livedoor_dict[key]:
            sample_text += key

print(sample_text)

