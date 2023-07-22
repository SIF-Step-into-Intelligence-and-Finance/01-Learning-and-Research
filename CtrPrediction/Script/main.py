import argparse
import os
import random
import sys
from datetime import datetime

from tqdm import tqdm

from data_iterator import DataIterator
from model import *
from utils import *

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
best_auc = 0.0  # 全局变量，记录训练过程中最好的测试结果:AUC
best_accuracy = 0.0  # 全局变量，记录训练过程中最好的测试结果:accuracy
no_improve_auc = 0
no_improve_accuracy = 0
trials_max = 20
end_flag = False  # 如果no_improve_auc>20或者no_improve_accuracy>20也即实验在20个batch内没有进展的话，我们视为过拟合，结束训练


def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()  # 获取变量的形状信息
        variable_parameters = shape.num_elements()  # 计算变量的参数数量
        total_parameters += variable_parameters  # 累加变量的参数数量
    return total_parameters


# 验证集就像测试集一样，所以这里并不需要反向传播，实际上也不需要aux_loss，但是我们这里也打印了出来
def eval(sess, test_data, model, best_model_path):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    result = []
    for src, click_label in tqdm(test_data, total=test_data.len(), ncols=80, desc='测试集'):
        nums += 1
        target, uids, mids, cats, lengths, mid_his, cat_his, his_tiv_target, his_tiv_neighbor, his_item_behaviors_list, his_item_behaviors_time_list, his_user_items_list, his_user_cats_list, mask_pad, mask_aux, mask_trend_for_history, mask_trend_for_target, noclk_mid_his, noclk_cat_his = prepare_data(src, click_label, args.max_sequence, return_neg=True)  # 这里是需要cpu计算的
        y_hat, loss, accuracy, aux_loss, summary = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, his_tiv_target, his_tiv_neighbor, his_item_behaviors_list, his_item_behaviors_time_list, mask_pad, mask_aux, mask_trend_for_history, mask_trend_for_target, lengths, target, False, his_user_items_list, his_user_cats_list, noclk_mid_his, noclk_cat_his])
        loss_sum += loss
        aux_loss_sum += aux_loss
        accuracy_sum += accuracy
        prob_1 = y_hat[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            result.append([p, t])
    test_auc = calc_auc(result)
    test_accuracy = accuracy_sum / nums
    test_loss = loss_sum / nums
    test_aux_loss = aux_loss_sum / nums
    global best_auc
    global no_improve_auc
    if best_auc < test_auc:
        print('best_auc is updating---------------------------------------------------------------------------------')
        sys.stdout.flush()
        best_auc = test_auc
        no_improve_auc = 0
        model.save(sess, best_model_path + args.train_model)
    else:
        no_improve_auc += 1  # 如果没有改善的话就加1，如果连续20次没改善，视为过拟合，下面会停止训练
    global best_accuracy
    global no_improve_accuracy
    if best_accuracy < test_accuracy:
        best_accuracy = test_accuracy
        no_improve_accuracy = 0
    else:
        no_improve_accuracy += 1  # 如果没有改善的话就加1，如果连续20次没改善，视为过拟合，下面会停止训练
    return {'test_auc': test_auc, 'test_loss': test_loss, 'test_accuracy': test_accuracy, 'test_aux_loss': test_aux_loss}, summary


def train():
    print('本次实验运行的参数为：', args)
    sys.stdout.flush()
    print(args, file=txt_log)
    txt_log.flush()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(args.train_file, args.uid_voc, args.mid_voc, args.cat_voc, args.tiv_voc, args.batch_size, args.max_sequence,shuffle_each_epoch=True,args=args)  # [[35228, 8417, 134, [13958, 9109, 1881, 15188], [2, 741, 134, 741], ['2', '2', '2', '1']]
        test_data = DataIterator(args.test_file, args.uid_voc, args.mid_voc, args.cat_voc, args.tiv_voc, args.batch_size, args.max_sequence,shuffle_each_epoch=False,args=args)
        n_uid, n_mid, n_cat, n_tiv = train_data.get_n()
        if args.train_model == 'Model_xDeepFM':
            model = Model_xDeepFM(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_FM':
            model = Model_FM(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_DeepFM':
            model = Model_DeepFM(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_WideDeep':
            model = Model_WideDeep(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_BST':
            model = Model_BST(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_DIN_New':
            model = Model_DIN_New(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_DIEN':
            model = Model_DIEN(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_DIEN1':
            model = Model_DIEN1(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'Model_DIEN2':
            model = Model_DIEN2(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'myModel_V2':
            model = myModel_V2(n_uid, n_mid, n_cat, n_tiv, args)
        elif args.train_model == 'myModel_V3':
            model = myModel_V3(n_uid, n_mid, n_cat, n_tiv, args)
        else:
            print("Invalid args.train_model : ", args.train_model)
            return
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        total_parameters = count_parameters()
        print("总的可训练参数量为: ", total_parameters)  # 打印总参数数量
        sys.stdout.flush()
        print("总的可训练参数量为: ", total_parameters, file=txt_log)
        txt_log.flush()

        summary_log = '../summary_log/' + args.train_model + '/' + TIMESTAMP + '_' + file_name
        train_writer = tf.summary.FileWriter(summary_log + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(summary_log + '/test', sess.graph)
        iter = -1
        lr = args.lr
        test_iter = args.test_iter
        for epoch in range(100):
            # 如果模型多次没有精度改进，就停止训练
            global end_flag
            if end_flag:
                global best_auc
                global best_accuracy
                print('best_auc: ', best_auc)
                print('best_accuracy: ', best_accuracy)
                break
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            for src, click_label in tqdm(train_data, total=train_data.len(), ncols=80, colour='green', desc='训练集'):
                # uids, mids, cats, mid_his, cat_his, his_tiv_target, his_tiv_neighbor, mask_pad, mask_aux, mask_trend_for_history, mask_trend_for_target, np.array(lengths_x), np.array(target), noclk_mid_his, noclk_cat_his
                target, uids, mids, cats, lengths, mid_his, cat_his, his_tiv_target, his_tiv_neighbor, his_item_behaviors_list, his_item_behaviors_time_list, his_user_items_list, his_user_cats_list, mask_pad, mask_aux, mask_trend_for_history, mask_trend_for_target, noclk_mid_his, noclk_cat_his = prepare_data(src, click_label, args.max_sequence, return_neg=True)  # 这里是需要cpu计算的
                loss, accuracy, aux_loss, summary = model.train(sess, [uids, mids, cats, mid_his, cat_his, his_tiv_target, his_tiv_neighbor, his_item_behaviors_list, his_item_behaviors_time_list, mask_pad, mask_aux, mask_trend_for_history, mask_trend_for_target, lengths, target, args.lr, True, his_user_items_list, his_user_cats_list, noclk_mid_his, noclk_cat_his])
                train_writer.add_summary(summary, iter)
                loss_sum += loss
                accuracy_sum += accuracy
                aux_loss_sum += aux_loss
                iter += 1
                if (iter % args.train_iter) == 0 and iter != 0:
                    print('epoch: ', epoch, '\t', 'iter: ', iter, '\t', 'train_loss: ', loss_sum / args.train_iter, '\t', 'train_accuracy: ', accuracy_sum / args.train_iter, '\t', 'train_aux_loss: ', aux_loss_sum / args.train_iter)
                    sys.stdout.flush()
                    print('epoch: ', epoch, '\t', 'iter: ', iter, '\t', 'train_loss: ', loss_sum / args.train_iter, '\t', 'train_accuracy: ', accuracy_sum / args.train_iter, '\t', 'train_aux_loss: ', aux_loss_sum / args.train_iter, file=txt_log)
                    txt_log.flush()
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % test_iter) == 0 and iter != 0:
                    metrics, summary = eval(sess, test_data, model, best_model_path)
                    test_writer.add_summary(summary, iter)
                    print('epoch: ', epoch, '\t', 'iter: ', iter, '\t', 'test_loss: ', metrics['test_loss'], '\t', 'test_auc: ', metrics['test_auc'], '\t', 'test_accuracy: ', '\t', metrics['test_accuracy'], '\t', 'train_aux_loss: ', metrics['test_aux_loss'])
                    sys.stdout.flush()
                    print('epoch: ', epoch, '\t', 'iter: ', iter, '\t', 'test_loss: ', metrics['test_loss'], '\t', 'test_auc: ', metrics['test_auc'], '\t', 'test_accuracy: ', '\t', metrics['test_accuracy'], '\t', 'train_aux_loss: ', metrics['test_aux_loss'], file=txt_log)
                    txt_log.flush()
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if test_iter != 200 and iter > 9000:  # 到后边的话测试频繁一点
                    test_iter = 200
                # if (iter % args.save_iter) == 0 and iter != 0:
                #     print('save model iter: %d' % (iter))
                #     model.save(sess, model_path + "--" + str(iter))
                global no_improve_auc
                global no_improve_accuracy
                if no_improve_auc >= trials_max or no_improve_accuracy >= trials_max:
                    end_flag = True
                    break
                lr = lr * args.lr_decay_rate ** (iter / args.lr_decay_steps)
        print('best_accuracy: ', best_accuracy, file=txt_log)
        print('best_auc: ', best_auc, file=txt_log)
        txt_log.flush()
        txt_log.close()
        train_writer.close()
        test_writer.close()
#更改数据集：
#1.历史行为序列的最大长度需要改（max_sequence）
#2.多久测试一下需要改（test_iter）
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_model', default="Model_WideDeep")
    # parser.add_argument('--train_model', default="Model_DNN")
    # parser.add_argument('--train_model', default="Model_DIN")
    # parser.add_argument('--train_model', default="Model_DIN_New")
    # parser.add_argument('--train_model', default="Model_DIEN")
    # parser.add_argument('--train_model', default="Model_DIEN2")
    # parser.add_argument('--train_model', default="Model_BST")
    # parser.add_argument('--train_model', default="myModel_V1")
    parser.add_argument('--train_model', default="myModel_V2")
    # parser.add_argument('--train_model', default="myModel_V3")
    parser.add_argument('--train_file', default="../dataset/Clothing/Clothing_train_V4_expand")
    parser.add_argument('--test_file', default="../dataset/Clothing/Clothing_test_V4_expand")
    parser.add_argument('--uid_voc', default="../dataset/Clothing/uid_voc_V4_expand.pkl")
    parser.add_argument('--mid_voc', default="../dataset/Clothing/mid_voc_V4_expand.pkl")
    parser.add_argument('--cat_voc', default="../dataset/Clothing/cat_voc_V4_expand.pkl")
    parser.add_argument('--tiv_voc', default="../dataset/Clothing/tiv_voc_V4_expand.pkl")
    parser.add_argument('--max_sequence', type=int, default=10)  # 用户历史行为序列的最大长度，Electronic设置为15，books设置为20
    parser.add_argument('--num_blocks', type=int, default=2)  # 用多少层masked_self_attention（设定为2比较合适）
    parser.add_argument('--train_iter', type=int, default=100)
    # ------------------------------这些训练输出是需要根据数据量调节的------------------------------
    parser.add_argument('--test_iter', type=int, default=200)
    parser.add_argument('--save_iter', type=int, default=200)
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--l2_reg', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)  # 0.1不行，太大了。学习率过大会导致rnn梯度爆炸：Infinity in summary histogram for: rnn_2/GRU_outputs2
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
    dir_name, file_name = os.path.split(args.train_file)
    txt_log_path = '../txt_log/' + TIMESTAMP + '_' + args.train_model + '_' + file_name
    txt_log = open(txt_log_path, 'w', encoding='utf-8')
    # ----------------------------运行模型之前一定在这里修改需要记录的重要信息---------------------------
    print('element-wise的激活函数改成relu，最后一层不要激活函数')
    sys.stdout.flush()
    print('element-wise的激活函数改成relu，最后一层不要激活函数', file=txt_log)
    txt_log.flush()
    model_path = "../save_path/" + args.train_model + '/' + TIMESTAMP + '_' + file_name
    if not os.path.exists(os.path.dirname(model_path)):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(os.path.dirname(model_path))
    best_model_path = "../best_model/" + args.train_model + '/' + TIMESTAMP + '/' + file_name
    if not os.path.exists(os.path.dirname(best_model_path)):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(os.path.dirname(best_model_path))
    gpu_options = tf.GPUOptions(allow_growth=True)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train()
