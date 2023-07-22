import argparse

from tqdm import tqdm
from utils import  prepare_data
from data_iterator import DataIterator

parser = argparse.ArgumentParser()
# parser.add_argument('--train_model', default="Model_WideDeep")
# parser.add_argument('--train_model', default="Model_DNN")
# parser.add_argument('--train_model', default="Model_DIN")
# parser.add_argument('--train_model', default="Model_DIEN")
# parser.add_argument('--train_model', default="Model_DIEN2")
# parser.add_argument('--train_model', default="Model_BST")
# parser.add_argument('--train_model', default="myModel_V1")
parser.add_argument('--train_model', default="myModel_V2")
# parser.add_argument('--train_model', default="myModel_V3")
parser.add_argument('--train_file', default="../dataset/Clothing/Clothing_train_V4_expand")
parser.add_argument('--test_file', default="../dataset/Clothing/Clothing_valid_V4_expand")
parser.add_argument('--uid_voc', default="../dataset/Clothing/uid_voc_V4_expand.pkl")
parser.add_argument('--mid_voc', default="../dataset/Clothing/mid_voc_V4_expand.pkl")
parser.add_argument('--cat_voc', default="../dataset/Clothing/cat_voc_V4_expand.pkl")
parser.add_argument('--tiv_voc', default="../dataset/Clothing/tiv_voc_V4_expand.pkl")
parser.add_argument('--max_sequence', type=int, default=10)  # 用户历史行为序列的最大长度，Clothing设置为10，其余的设置为20
parser.add_argument('--num_blocks', type=int, default=2)  # 用多少层masked_self_attention（设定为2比较合适）
parser.add_argument('--train_iter', type=int, default=100)
# ------------------------------这些训练输出是需要根据数据量调节的------------------------------
parser.add_argument('--test_iter', type=int, default=200)
parser.add_argument('--save_iter', type=int, default=200)
# ---------------------------------------------------------------------------------------
parser.add_argument('--l2_reg', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.01)  # 学习率过大会导致rnn梯度爆炸：Infinity in summary histogram for: rnn_2/GRU_outputs2
parser.add_argument('--lr_decay_steps', type=int, default=10000)
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=3)
# ---------------------------注意以下这些维度如果做交互的话，需要统一一下---------------------------
parser.add_argument('--user_emb_dim', type=int, default=36)
parser.add_argument('--item_emb_dim', type=int, default=24)
parser.add_argument('--cat_emb_dim', type=int, default=12)
parser.add_argument('--position_emb_dim', type=int, default=4)  # 物品位置embedding
parser.add_argument('--tiv_emb_dim', type=int, default=4)  # 时间间隔embedding
parser.add_argument('--rnn_hidden_dim', type=int, default=36)
parser.add_argument('--all_candidate_trends', type=int, default=1000)
parser.add_argument('--item_trends', type=int, default=5)
args = parser.parse_args()

train_data = DataIterator(args.train_file, args.uid_voc, args.mid_voc, args.cat_voc, args.tiv_voc, args.batch_size, args.max_sequence)

for src, click_label in tqdm(train_data, total=train_data.len(), ncols=80, colour='green', desc='训练集'):
    target, uids, mids, cats, lengths, mid_his, cat_his, his_tiv_target, his_tiv_neighbor, his_item_behaviors_list, his_item_behaviors_time_list, his_user_items_list, his_user_cats_list, mask_pad, mask_aux, mask_trend_for_history, mask_trend_for_target, noclk_mid_his, noclk_cat_his = prepare_data(src, click_label, args.max_sequence, return_neg=True)
    print(his_user_items_list)
    print(his_user_cats_list)
