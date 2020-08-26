from hmmlearn import hmm
from text_analysis import TextAnalysis

input_dir = "./datasets/livedoor/dokujo-tsushin/"
input_data = input_dir + "dokujo-tsushin-4778030.txt"

X = TextAnalysis.mecab_analysis(input_data)
print(X)

# verbose=Trueで各回のイテレーションを確認できる．
# model = hmm.MultinomialHMM(n_components=10, n_iter=1000, verbose=True)
model = hmm.MultinomialHMM(n_components=10, n_iter=1000)

model.fit(X)

L,Z = model.decode(X)
# print(model.transmat_) # 遷移確率の出力 
# print(model.monitor_) # historyの配列は最後から2つの対数尤度を出力している．