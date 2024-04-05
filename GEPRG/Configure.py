train_data_path = "../data/no_cycle/train.data"
dev_data_path = "../data/no_cycle/dev.data"
test_data_path = "../data/no_cycle/test.data"
word_idx_file_path = "../data/word.idx"

train_batch_size = 5  #训练每轮数据量
dev_batch_size = 5 #推理每轮数据量
test_batch_size = 5 #测试每轮数据量
l2_lambda = 0.000001 #正则化？
learning_rate = 0.001 #学习率
epochs = 100 #轮数

unknown_word = "<unk>"
PAD = "<PAD>"
GO = "<GO>"
EOS = "<EOS>"
deal_unknown_words = True

## GCN ############################################
word_size_max = 10 #特征最大词数
hidden_layer_dim = 1000  #特征嵌入向量维数大小
# graph_encode_method = "max-pooling" # "lstm" or "max-pooling"
sample_layer_size = 4 #聚合前后几个节点的信息（采样后）
sample_size_per_layer = 4 #邻居节点采样大小
concat = True #聚合前后节点信息时是否拼接
graph_encode_direction = "bi" # "single" or "bi" 双向节点聚合

## RNN #############################################
dropout = 0.0 #encode cell或者decode cell中训练时需要dropout
num_layers = 1 # 1 or 2 单个解码器/编码器单元basic模块层数
attention = True
decoder_type = "greedy" # greedy, beam  测试时采用序列生成方法
beam_width = 4
seq_max_len = 11 #decode最长生成词数

path_embed_method = "lstm" # cnn or lstm or bi-lstm  特征编码/编码器单元/解码器单元/图嵌入编码生成

## 无用 ###############################################
feature_max_len = 1 #特征最大词数（已存在
word_embedding_dim = 100  #特征嵌入向量维数大小（已存在
num_layers_decode = 1 #单个解码器单元basic模块层数（已存在

feature_encode_type = "uni"
encoder = "gated_gcn" # "gated_gcn" "gcn"  "seq" 编码器解码器中提取
lstm_in_gcn = "none" # before, after, none
encoder_hidden_dim = 200
weight_decay = 0.0000