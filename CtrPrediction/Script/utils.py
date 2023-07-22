import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *

from core_rnn_cell import _Linear


def prepare_data(input, click_labels, maxlen=None, return_neg=True):  # 这里每一个批次的最大长度不一样，所以用dynamic_rnn
    # input=[[uid, mid, cat,target_time, mid_list, cat_list,  hitory_time_list, tiv_target_list, tiv_neighbor_list, noclk_mid_list, noclk_cat_list],[...],...]
    uids = np.array([inp[0] for inp in input])  # batch_size个uers_id[128]
    mids = np.array([inp[1] for inp in input])  # batch_size个目标物品id[128]
    cats = np.array([inp[2] for inp in input])  # batch_size个目标物品种类id[128]
    target_time = np.array([int(inp[3]) for inp in input])  # batch_size个目标物品时间[128]
    mask_trend_for_target = (target_time >= 1397116033).astype(np.float32)  # 这里计算物品流行度的时候需不需要目标物品来计算[128]
    lengths_x = [len(inp[4]) for inp in input]  # 每个样本的历史交互序列长度（注意这里实际上我们默认了mid_lsit,cat_list, tiv_target_list,tiv_neighbor_list, noclk_mid_list, noclk_cat_list）都是一样的长度，所以选择了mid_list的长度
    seqs_mid = [inp[4] for inp in input]  # 列表的列表：用户的历史交互的物品id[128,5]
    seqs_cat = [inp[5] for inp in input]  # 列表的列表：用户的历史交互的物品种类id[128,5]
    seqs_history_time = [inp[6] for inp in input]  # 列表的列表：用户的历史交互的时间戳[128,5]
    seqs_tiv_target = [inp[7] for inp in input]  # 列表的列表：用户的历史交互的物品距目标物品的时间间隔[128,5]
    seqs_tiv_neighbor = [inp[8] for inp in input]  # 列表的列表：用户的历史交互的物品距下一个物品的时间间隔[128,5]
    item_behaviors_list = [inp[9] for inp in input]  # 列表的列表的列表：历史物品对应的物品行为序列。DataIterator里我们已经将第二维填充成一样的[128,5,3]
    item_behaviors_tiv_list = [inp[10] for inp in input]  # 列表的列表的列表：[128,5,3]
    noclk_seqs_mid = [inp[11] for inp in input]  # 列表的列表的列表：[128,5,5]
    noclk_seqs_cat = [inp[12] for inp in input]  # 列表的列表的列表：[128,5,5]
    history_user_items_list = [inp[13] for inp in input]  # 列表的列表的列表的列表：用户购买的物品序列，对应的曾经购买用户序列，曾经购买用户序列对应的物品序列[128,5,3,3]
    history_user_cats_list = [inp[14] for inp in input]  # 列表的列表的列表的列表：[128,5,3,3]
    # history_user_times_list = [inp[15] for inp in input]
    # ------------------------------ 历史行为序列长度需要截断（样本文件中可能已经截断，这里再截断一下，为了方便测试不同的序列长度） ----------------------------------
    if maxlen is not None:
        for i, (l_x, inp) in enumerate(zip(lengths_x, input)):
            if l_x > maxlen:
                lengths_x[i] = maxlen
                seqs_mid[i] = seqs_mid[i][- maxlen:]
                seqs_cat[i] = seqs_cat[i][- maxlen:]
                seqs_history_time[i] = seqs_history_time[i][- maxlen:]
                seqs_tiv_target[i] = seqs_tiv_target[i][- maxlen:]
                seqs_tiv_neighbor[i] = seqs_tiv_neighbor[i][- maxlen:]
                item_behaviors_list[i] = item_behaviors_list[i][- maxlen:]
                item_behaviors_tiv_list[i] = item_behaviors_tiv_list[i][- maxlen:]
                noclk_seqs_mid[i] = noclk_seqs_mid[i][- maxlen:]
                noclk_seqs_cat[i] = noclk_seqs_cat[i][- maxlen:]
                history_user_items_list = history_user_items_list[i][- maxlen:]
                history_user_cats_list = history_user_cats_list[i][- maxlen:]  # history_user_times_list = history_user_times_list[i][- maxlen:]
    # ------------------------------ 需要填充，这里填充的第二维度（序列长度）----------------------------------
    n_samples = len(seqs_mid)  # 有多少个样本:128
    # maxlen_x = np.max(lengths_x)  # 每批样本中行为序列的最长长度
    maxlen_x = 10  # 手动指定每批样本中行为序列的最长长度
    neg_samples = len(noclk_seqs_mid[0][0])  # 每个行为对应的负样本的数量
    his_users_num = len(history_user_items_list[0][0])  # 用户购买的物品序列，对应的曾经购买用户序列的长度
    his_user_items_num = len(history_user_items_list[0][0][0])  # 用户购买的物品序列，对应的曾经购买用户序列的长度，曾经购买用户序列对应的物品序列的长度

    item_behaviors_max_length = max(len(sublist) for sublist1 in item_behaviors_list for sublist in sublist1)  # 每个物品行为序列的长度最大值
    mask_pad = np.zeros((n_samples, maxlen_x), dtype=np.float32)
    mask_aux = np.zeros((n_samples, maxlen_x), dtype=np.float32)
    mask_trend_for_history = np.zeros((n_samples, maxlen_x), dtype=np.float32)

    mid_his = np.zeros((n_samples, maxlen_x), dtype=np.int64)
    cat_his = np.zeros((n_samples, maxlen_x), dtype=np.int64)
    his_time = np.zeros((n_samples, maxlen_x), dtype=np.float32)  # 填充的时间都时0，所以下面的mask_trend掩盖的实际上大于mask_pad
    his_tiv_target = np.zeros((n_samples, maxlen_x), dtype=np.int64)
    his_tiv_neighbor = np.full((n_samples, maxlen_x), 2, dtype=np.int64)  # 这里取2，因为下面需要将小于等于1的置为1
    his_item_behaviors_list = np.zeros((n_samples, maxlen_x, item_behaviors_max_length), dtype=np.int64)  # 默认的用户id(default)为0
    his_item_behaviors_tiv_list = np.full((n_samples, maxlen_x, item_behaviors_max_length), 18, dtype=np.int64)  # 默认的最长的时间间隔为18
    noclk_mid_his = np.zeros((n_samples, maxlen_x, neg_samples), dtype=np.int64)
    noclk_cat_his = np.zeros((n_samples, maxlen_x, neg_samples), dtype=np.int64)
    his_user_items_list = np.zeros((n_samples, maxlen_x, his_users_num, his_user_items_num), dtype=np.int64)  # 默认的物品为0
    his_user_cats_list = np.zeros((n_samples, maxlen_x, his_users_num, his_user_items_num), dtype=np.int64)  # 默认的类别为0
    # his_user_times_list = np.zeros((n_samples, maxlen_x, his_users_num, his_user_items_num), dtype=np.int64)

    for idx, [s_x, s_y, history_time, tiv_target, tiv_neighbor, item_beh_list, item_beh_time_list, no_s_x, no_s_y, his_user_items_list_temp, his_user_cats_list_temp] in enumerate(zip(seqs_mid, seqs_cat, seqs_history_time, seqs_tiv_target, seqs_tiv_neighbor, item_behaviors_list, item_behaviors_tiv_list, noclk_seqs_mid, noclk_seqs_cat, history_user_items_list, history_user_cats_list)):
        mask_pad[idx, :lengths_x[idx]] = 1.  # mask的作用：标志前边都是原有的历史交互物品id，后边的都是填充的
        mask_aux[idx] = np.where(his_tiv_neighbor[idx] <= 1, 1., 0.)
        mask_trend_for_history[idx] = np.where(his_time[idx] >= 0, 1., 0.)  # 这里计算物品流行度的时候需要较为近期的（绝对时间）来计算[128,5]。当>0的时候，说明所有的历史物品都算上
        mid_his[idx, :lengths_x[idx]] = s_x  # 填充后的数据，lengths_x[idx]之前的是原有的数据，后边的都是0
        cat_his[idx, :lengths_x[idx]] = s_y  # 填充后的数据，lengths_x[idx]之前的是原有的数据，后边的都是0
        his_time[idx, :lengths_x[idx]] = history_time
        his_tiv_target[idx, :lengths_x[idx]] = tiv_target
        his_tiv_neighbor[idx, :lengths_x[idx]] = tiv_neighbor
        his_item_behaviors_list[idx, :lengths_x[idx], :] = item_beh_list
        his_item_behaviors_tiv_list[idx, :lengths_x[idx], :] = item_beh_time_list
        noclk_mid_his[idx, :lengths_x[idx], :] = no_s_x
        noclk_cat_his[idx, :lengths_x[idx], :] = no_s_y
        his_user_items_list[idx, :lengths_x[idx], :, :] = his_user_items_list_temp
        his_user_cats_list[idx, :lengths_x[idx], :, :] = his_user_cats_list_temp  # his_user_times_list[idx, :lengths_x[idx], :, :] = his_user_times_list_temp

    if not return_neg:
        return np.array(click_labels), uids, mids, cats, np.array(lengths_x), mid_his, cat_his, his_tiv_target, his_tiv_neighbor, his_item_behaviors_list, his_item_behaviors_tiv_list, his_user_items_list, his_user_cats_list, mask_pad, mask_aux, mask_trend_for_history, mask_trend_for_target,
    else:
        return np.array(click_labels), uids, mids, cats, np.array(lengths_x), mid_his, cat_his, his_tiv_target, his_tiv_neighbor, his_item_behaviors_list, his_item_behaviors_tiv_list, his_user_items_list, his_user_cats_list, mask_pad, mask_aux, mask_trend_for_history, mask_trend_for_target, noclk_mid_his, noclk_cat_his


class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear([inputs, state], 2 * self._num_units, True, bias_initializer=bias_ones, kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear([inputs, r_state], self._num_units, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h


class QAAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(QAAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear([inputs, state], 2 * self._num_units, True, bias_initializer=bias_ones, kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear([inputs, r_state], self._num_units, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = (1. - att_score) * state + att_score * c
        return new_h, new_h


def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1], dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc


def dynamic_item_attention_with_his_user(user_id_emb, his_item_behaviors_emb, padding_mask=None, softmax_stag=True):  # [128,36],[128,5,3,36],[128,5,3]
    user_id_embs = tf.tile(user_id_emb, [1, tf.shape(his_item_behaviors_emb)[1] * tf.shape(his_item_behaviors_emb)[2]])  # [128,5*3*36]
    user_id_embs = tf.reshape(user_id_embs, tf.shape(his_item_behaviors_emb))  # [128,5,3,36]
    inputs_mlp = tf.concat([user_id_embs, his_item_behaviors_emb, user_id_embs - his_item_behaviors_emb, user_id_embs * his_item_behaviors_emb], axis=-1)
    hidden_1 = tf.layers.dense(inputs_mlp, 64, activation=tf.nn.sigmoid, name='dynamic_item_attention_with_his_user_hidden_1')
    hidden_2 = tf.layers.dense(hidden_1, 32, activation=tf.nn.sigmoid, name='dynamic_item_attention_with_his_user_hidden_2')
    hidden_3 = tf.layers.dense(hidden_2, 1, activation=None, name='dynamic_item_attention_with_his_user_hidden_3')  # [128,5,3,1]
    scores = tf.squeeze(hidden_3, -1)  # [128,5,3,1]->[128,5,3]
    if padding_mask is not None:  # [128,5,3]
        padding_mask = tf.cast(padding_mask, tf.bool)
        paddings1 = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(padding_mask, scores, paddings1)
    if softmax_stag:
        scores = tf.nn.softmax(scores)
    output = his_item_behaviors_emb * tf.expand_dims(scores, -1)  # [128,5,3,36]*[128,5,3,1]->[128,5,3,36]
    output = tf.reduce_sum(output, 2)  # [128,5,3,36]->[128,5,36]
    return output, scores  # [128,5,36],[128,5,3]


def dynamic_item_attention_with_his_user_items(history_items_emb, his_user_items_emb, padding_mask=None, softmax_stag=True):  # [128,36],[128,5,3,36],[128,5,3]
    his_user_items_emb = tf.reshape(his_user_items_emb, [128, -1, 9, 36])  # [128,5,3,3,36]->[128,5,9,36]
    print(his_user_items_emb)
    history_items_embs = tf.tile(history_items_emb, [1, 1, tf.shape(his_user_items_emb)[2]])  # [128,5,36] -> [128,5,9*36]
    history_items_embs = tf.reshape(history_items_embs, tf.shape(his_user_items_emb))  # [128,5,9*36] -> [128,5,9,36]

    inputs_mlp = tf.concat([history_items_embs, his_user_items_emb, history_items_embs - his_user_items_emb, history_items_embs * his_user_items_emb], axis=-1)  # [128,5,9,36+36+36+36]
    hidden_1 = tf.layers.dense(inputs_mlp, 64, activation=tf.nn.sigmoid, name='dynamic_item_attention_with_his_user_items_hidden_1')
    hidden_2 = tf.layers.dense(hidden_1, 32, activation=tf.nn.sigmoid, name='dynamic_item_attention_with_his_user_items_hidden_2')
    hidden_3 = tf.layers.dense(hidden_2, 1, activation=None, name='dynamic_item_attention_with_his_user_items_hidden_3')  # [128,5,3,1]
    scores = tf.squeeze(hidden_3, -1)  # [128,5,9,1]->[128,5,9]
    if padding_mask is not None:  # [128,5,9]
        padding_mask = tf.cast(padding_mask, tf.bool)
        paddings1 = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(padding_mask, scores, paddings1)
    if softmax_stag:
        scores = tf.nn.softmax(scores)
    output = his_user_items_emb * tf.expand_dims(scores, -1)  # [128,5,9,36]*[128,5,9,1]->[128,5,9,36]
    output = tf.reduce_sum(output, 2)  # [128,5,9,36]->[128,5,36]
    return output, scores  # [128,5,36],[128,5,9]


def element_wise_attention(transposed_history_items_emb, target_item_emb, user_emb):  # [128,36,5],[128,36],[128,36]
    #这里采取的是将每一个维度上的特征都与target_item_emb和user_emb拼接起来，输入一个神经网络。但是这样的话，各个维度的建模就分开了，并且无法区分输入是哪个维度。
    hidden_1 = tf.layers.dense(tf.concat([transposed_history_items_emb, tf.tile(tf.expand_dims(target_item_emb, 1), [1, tf.shape(transposed_history_items_emb)[1], 1]), tf.tile(tf.expand_dims(user_emb, 1), [1, tf.shape(user_emb)[1], 1])], -1), 32, activation=tf.nn.tanh, name='element_wise_attention_hidden_1')
    hidden_2 = tf.layers.dense(hidden_1, 16, activation=tf.nn.tanh, name='element_wise_attention_hidden_2')
    hidden_3 = tf.layers.dense(hidden_2, 1, activation=None, name='element_wise_attention_hidden_3')
    scores = tf.reshape(hidden_3, [-1, 1, tf.shape(transposed_history_items_emb)[1]])  # [128,36,1]->[128,1,36]
    softmax_scores = tf.nn.softmax(scores)
    return softmax_scores  # [128,1,36]


def din_fcn_attention(query, keys, padding_mask, softmax_stag=True, din=False):  # [128,36],[128,5,36],[128,5]
    padding_mask = tf.cast(padding_mask, tf.bool)
    if not din:
        keys_size = keys.get_shape().as_list()[-1]  # 36
        query = tf.layers.dense(query, keys_size, activation=None, name='din_fcn_attention_f1')  # [128,36]
        query = prelu(query)  # [128,36]
    queries = tf.tile(query, [1, tf.shape(keys)[1]])  # [128,5*36]
    queries = tf.reshape(queries, tf.shape(keys))  # [128,5,36]

    hidden_3=tf.reduce_sum(queries*keys,-1)
    # inputs_mlp = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    # hidden_1 = tf.layers.dense(inputs_mlp, 80, activation=tf.nn.sigmoid, name='din_fcn_attention_hidden_1')
    # hidden_2 = tf.layers.dense(hidden_1, 40, activation=tf.nn.sigmoid, name='din_fcn_attention_hidden_2')
    # hidden_3 = tf.layers.dense(hidden_2, 1, activation=None, name='din_fcn_attention_hidden_3')
    scores = tf.reshape(hidden_3, [-1, 1, tf.shape(keys)[1]])  # [128,5,1]->[128,1,5]
    key_masks = tf.expand_dims(padding_mask, 1)  # [128,5]->[128,1,5]
    paddings1 = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings1)
    if softmax_stag:
        scores = tf.nn.softmax(scores)
    scores = tf.reshape(scores, tf.shape(padding_mask))  # [128,5]
    output = keys * tf.expand_dims(scores, -1)  # [128,5,36]*[128,5,1]->[128,5,36]
    output = tf.reshape(output, tf.shape(keys))
    return output, scores  # [128,5,36],[128,5]


def dien_fcn_attention(query, keys, padding_mask, din=False, causality=True):  # 注意填充掩码和因果掩码 [128,5,36],[128,5,36],[128,5]
    padding_mask = tf.cast(padding_mask, tf.bool)
    if not din:  # 如果不是din，比如说DIEN，说明query跟key不在一个空间中（有可能维度不一样），需要先做一个变换
        keys_size = keys.get_shape().as_list()[-1]  # 36
        query = tf.layers.dense(query, keys_size, activation=None, name='dien_fcn_attention_f1')  # [128,5,36]
    queries = tf.tile(tf.expand_dims(query, 2), [1, 1, tf.shape(keys)[1], 1])  # [2,3,3,2]#[batch,seq_len,seq_len,emb]
    keys = tf.tile(tf.expand_dims(keys, 2), [1, 1, tf.shape(keys)[1], 1])  # [2,3,3,2]#[batch,seq_len,seq_len,emb]
    keys = tf.transpose(keys, [0, 2, 1, 3])  # [2,3,3,2]#[seq_len,batch,seq_len,emb]后两个维度是指将embedding重复
    inputs_mlp = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # [2,3,3,2]
    hidden_1 = tf.layers.dense(inputs_mlp, 80, activation=tf.nn.sigmoid, name='dien_fcn_attention_hidden_1')
    hidden_2 = tf.layers.dense(hidden_1, 40, activation=tf.nn.sigmoid, name='dien_fcn_attention_hidden_2')
    scores = tf.layers.dense(hidden_2, 1, activation=None, name='dien_fcn_attention_hidden_3')  # [2,3,3,1]
    key_masks = tf.expand_dims(padding_mask, -1)  # [2,3]->[2,3,1]
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[2], 1, 1])  # [2,3,1]->[2,3,3,1]最后两维度就是（填充的兴趣）
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [2,3,3,1]
    if causality:
        diagonal_matrix = tf.ones_like(scores[0, :, :, 0])  # (3,3)
        triangle_lower_matrix = tf.linalg.LinearOperatorLowerTriangular(diagonal_matrix).to_dense()  # (3, 3)#转换成下三角矩阵
        masks = tf.reshape(tf.tile(tf.expand_dims(triangle_lower_matrix, 0), [tf.shape(scores)[0], 1, 1]), tf.shape(scores))  # (2,3,3,1)
        scores = tf.where(tf.equal(masks, 0), paddings, scores)  # [2,3,3,1]
    scores = tf.nn.softmax(scores, axis=2)  # [2,3,3,1]
    output1 = keys * scores  ##[2,3,3,2]*#[2,3,3,1]->[2,3,3,2]
    output = tf.reduce_sum(output1, 2)
    return output, tf.squeeze(scores)  # [5,128,36],[5,128,5]#这里scores的形状是不确定的，因为我们句子的长度不确定，有可能是5,6,。。。20


def attention(query, keys, padding_mask, mode=None):  # query:[128,36],keys=[128,5,9]，返回的跟keys的形状是一样的（没加和），或者加和
    keys = tf.concat(keys, -1)  # keys传进来的是一个列表，列表中的张量形状除了最后一维要相同
    padding_mask = tf.cast(padding_mask, tf.bool)  # [128,5]
    queries = tf.tile(tf.expand_dims(query, 1), [1, tf.shape(keys)[1], 1])  # [128,5,36]
    # ---------------------------------query跟key的形状不一定一样，先转换一下，方便下面做queries - keys, queries * keys------------------------------------
    queries = tf.layers.dense(queries, keys.get_shape().as_list()[-1], activation=None)  # 这里对于每个兴趣头，生成相应的表示
    inputs_mlp = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # [128,5,96]
    hidden_1 = tf.layers.dense(inputs_mlp, 80, activation=tf.nn.tanh, name='f1_att')  # [128,5,80]
    hidden_2 = tf.layers.dense(hidden_1, 40, activation=tf.nn.tanh, name='f2_att')  # [128,5,40]
    hidden_3 = tf.layers.dense(hidden_2, 1, activation=None, name='f3_att')  # [128,5,1]
    scores = tf.reshape(hidden_3, [-1, tf.shape(keys)[1]])  # [128,5]
    # padding mask
    paddings1 = tf.ones_like(scores) * (-2 ** 32 + 1)  # [128,5]
    scores = tf.where(padding_mask, scores, paddings1)  # [128,5]
    paddings2 = tf.zeros_like(scores)  # [128,5]
    return_scores = tf.where(padding_mask, scores, paddings2)  # [128,5]
    softmax_scores = tf.nn.softmax(scores, axis=-1)  # [128 ,5]
    outputs = keys * tf.expand_dims(softmax_scores, -1)  # [128,5,9]*[128,5,1]->[128,5,9]
    if mode == 'SUM':
        return tf.reduce_sum(outputs, 1), return_scores  # [128,9], [128,5]
    else:
        return outputs, return_scores  # [128,5,9]


# 分开返回注意力机制的多头结果
def masked_multihead_attention_split(queries, keys, padding_mask, num_units=None, num_heads=4, dropout_rate=0.5, is_training=True, causality=True):
    if num_units is None:
        num_units = queries.get_shape().as_list()[-1]
    # Linear projections
    Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (128, 5, 144)
    K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (128, 5, 144)
    V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (128, 5, 144)
    # Split and concat多头注意力。注意这里先split后concat的操作（之后的计算是多个头并行计算，将其放在batch方向上，因为batch上就是并行计算）
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (512, 5, 36)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (512, 5, 36)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (512, 5, 36)
    # Multiplication
    scores = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (512, 5, 5)实际上如果不并行计算的话，我们得到的是4个[128,5,5]的得分，其中[128,1,5]这个形状的是每个query对keys的得分
    # Scale
    scores = scores / (math.sqrt(K_.get_shape().as_list()[-1]))  # (512, 5, 5)
    padding_mask = tf.cast(padding_mask, tf.bool)  # [128,5]
    # padding mask:Key Masking。query（其中包括不需要计算的query，下面再做query Masking）对每一个key屏蔽不需要计算的key的评分。如果加上了因果的话其实这里就是多余的
    key_masks = tf.tile(padding_mask, [num_heads, 1])  # [128,5]->[512,5]这里变换是指对于个头需要mask掉的东西是一样的
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # [512, 5, 5]这里变换是指对于每一个key需要mask掉的东西是一样的
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)  # 非常小的负数，softmax后几乎为0 # (512, 5, 5)
    scores = tf.where(key_masks, scores, paddings)  # (512, 5, 5)

    # 后面的行为不能影响前面的行为（模拟RNN）
    if causality:
        diagonal_matrix = tf.ones_like(scores[0, :, :])  # (5, 5)
        triangle_lower_matrix = tf.linalg.LinearOperatorLowerTriangular(diagonal_matrix).to_dense()  # (5, 5)#转换成下三角矩阵
        masks = tf.tile(tf.expand_dims(triangle_lower_matrix, 0), [tf.shape(scores)[0], 1, 1])  # (512, 5, 5)
        scores = tf.where(tf.equal(masks, 0), paddings, scores)  # (512, 5, 5)

    # Activation
    scores = tf.nn.softmax(scores)  # (512, 5, 5)
    # padding mask:Query Masking，这里是为了去除上面计算的冗余的query
    query_masks = tf.tile(padding_mask, [num_heads, 1])  # (512, 5)
    query_masks = tf.tile(tf.expand_dims(query_masks, 2), [1, 1, tf.shape(keys)[1]])  # (512, 5, 5)
    query_masks = tf.cast(query_masks, tf.float32)
    scores *= query_masks  # (512, 5, 5)
    # Dropouts
    # scores = tf.layers.dropout(scores, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    # Weighted sum
    outputs = tf.matmul(scores, V_)  # (512, 5, 5)*(512, 5, 36)->(512, 5, 36)

    # Restore shape多头计算完后，不要合并，直接返回分别与目标物品求attention
    outputs = tf.split(outputs, num_heads, axis=0)  # [4,128,5,36]:[(128,5,36),(128,5,36),(128,5,36),(128,5,36)]

    # Residual connection
    # outputs += queries  # [4,128,5,36]+(128,5,36)->[4,128,5,36]。广播技术

    return outputs


# 直接返回合并后的注意力机制的结果
def masked_multihead_attention_unite(queries, keys, padding_mask, num_units=None, num_heads=4, dropout_rate=0.5, is_training=True, causality=True):
    if num_units is None:
        num_units = queries.get_shape().as_list()[-1]
    # Linear projections
    Q = tf.layers.dense(queries, num_units, activation=tf.nn.tanh)  # (128, 5, 36)
    K = tf.layers.dense(keys, num_units, activation=tf.nn.tanh)  # (128, 5, 36)
    V = tf.layers.dense(keys, num_units, activation=tf.nn.tanh)  # (128, 5, 36)
    # Split and concat多头注意力。注意这里先split后concat的操作（之后的计算是多个头并行计算，将其放在batch方向上，因为batch上就是并行计算）
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (512, 5, 9)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (512, 5, 9)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (512, 5, 9)
    # Multiplication
    scores = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (512, 5, 5)实际上如果不并行计算的话，我们得到的是4个[128,5,5]的得分，其中[128,1,5]这个形状的是每个query对keys的得分
    # Scale
    scores = scores / (math.sqrt(K_.get_shape().as_list()[-1]))  # (512, 5, 5)
    padding_mask = tf.cast(padding_mask, tf.bool)  # [128,5]
    # padding mask:Key Masking。query（其中包括不需要计算的query，下面再做query Masking）对每一个key屏蔽不需要计算的key的评分。如果加上了因果的话其实这里就是多余的
    key_masks = tf.tile(padding_mask, [num_heads, 1])  # [128,5]->[512,5]这里变换是指对于个头需要mask掉的东西是一样的
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # [512, 5, 5]这里变换是指对于每一个key需要mask掉的东西是一样的
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)  # 非常小的负数，softmax后几乎为0 # (512, 5, 5)
    scores = tf.where(key_masks, scores, paddings)  # (512, 5, 5)
    # 后面的行为不能影响前面的行为（模拟RNN）
    if causality:
        diagonal_matrix = tf.ones_like(scores[0, :, :])  # (5, 5)
        triangle_lower_matrix = tf.linalg.LinearOperatorLowerTriangular(diagonal_matrix).to_dense()  # (5, 5)#转换成下三角矩阵
        masks = tf.tile(tf.expand_dims(triangle_lower_matrix, 0), [tf.shape(scores)[0], 1, 1])  # (512, 5, 5)
        scores = tf.where(tf.equal(masks, 0), paddings, scores)  # (512, 5, 5)
    # Activation
    scores = tf.nn.softmax(scores)  # (512, 5, 5)
    # padding mask:Query Masking，这里是为了去除上面计算的冗余的query
    query_masks = tf.tile(padding_mask, [num_heads, 1])  # (512, 5)
    query_masks = tf.tile(tf.expand_dims(query_masks, 2), [1, 1, tf.shape(keys)[1]])  # (512, 5, 5)
    query_masks = tf.cast(query_masks, tf.float32)
    scores *= query_masks  # (512, 5, 5)
    # Dropouts
    # scores = tf.layers.dropout(scores, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    # Weighted sum
    outputs = tf.matmul(scores, V_)  # (512, 5, 5)*(512, 5, 9)->(512, 5, 9)

    # Restore shape多头计算完后
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)  # (512, 5, 9)->[(128,5,9),(128,5,9),(128,5,9),(128,5,9)]->(128,5,36)

    # Residual connection
    outputs += Q  # (128,5,36)
    return outputs


def masked_multihead_attention_unite_V2(queries, keys, padding_mask, num_units=None, num_heads=4, dropout_rate=0.5, is_training=True, causality=True):
    if num_units is None:
        num_units = queries.get_shape().as_list()[-1]
    # Linear projections，这里实际上也是引入了不对称性
    Q = tf.layers.dense(queries, num_units, activation=tf.nn.tanh)  # (128, 5, 36)
    K = tf.layers.dense(keys, num_units, activation=tf.nn.tanh)  # (128, 5, 36)
    V = tf.layers.dense(keys, num_units, activation=tf.nn.tanh)  # (128, 5, 36)
    # Split and concat多头注意力。注意这里先split后concat的操作（之后的计算是多个头并行计算，将其放在batch方向上，因为batch上就是并行计算）
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (512, 5, 9)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (512, 5, 9)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (512, 5, 9)
    # Multiplication
    scores = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (512, 5, 5)实际上如果不并行计算的话，我们得到的是4个[128,5,5]的得分，其中[128,1,5]这个形状的是每个query对keys的得分
    # Scale
    scores = scores / (math.sqrt(K_.get_shape().as_list()[-1]))  # (512, 5, 5)
    padding_mask = tf.cast(padding_mask, tf.bool)  # [128,5]
    # padding mask:Key Masking。query（其中包括不需要计算（填充的）的query，下面再做query Masking）对每一个key屏蔽不需要计算的key的评分。如果加上了因果的话其实这里就是多余的
    key_masks = tf.tile(padding_mask, [num_heads, 1])  # [128,5]->[512,5]这里变换是指对于个头需要mask掉的东西是一样的
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # [512, 5, 5]这里变换是指对于每一个key需要mask掉的东西是一样的（也即填充的部分）
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)  # 非常小的负数，softmax后几乎为0 # (512, 5, 5)
    scores = tf.where(key_masks, scores, paddings)  # (512, 5, 5)

    # 后面的行为不能影响前面的行为（模拟RNN）
    if causality:
        diagonal_matrix = tf.ones_like(scores[0, :, :])  # (5, 5)
        diagonal_matrix = tf.linalg.set_diag(diagonal_matrix, tf.zeros(shape=tf.shape(scores)[1]))  # 将对角线元素设置为0，也就是说算的分的时候不算本身，为了防止本身所占比重太大而导致其它分数的难学习，但是下面直接加上了本身
        triangle_lower_matrix = tf.linalg.LinearOperatorLowerTriangular(diagonal_matrix).to_dense()  # (5, 5)#转换成下三角矩阵
        masks = tf.tile(tf.expand_dims(triangle_lower_matrix, 0), [tf.shape(scores)[0], 1, 1])  # (512, 5, 5)
        scores = tf.where(tf.equal(masks, 0), paddings, scores)  # (512, 5, 5)

    # Activation
    scores = tf.nn.softmax(scores)  # (512, 5, 5)
    # padding mask:Query Masking，这里是为了去除上面计算的冗余的query（填充的一整行都需要去除）
    query_masks = tf.tile(padding_mask, [num_heads, 1])  # (512, 5)
    query_masks = tf.tile(tf.expand_dims(query_masks, 2), [1, 1, tf.shape(keys)[1]])  # (512, 5, 5)
    query_masks = tf.cast(query_masks, tf.float32)
    scores *= query_masks  # (512, 5, 5)
    # Dropouts
    # scores = tf.layers.dropout(scores, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    # Weighted sum
    # outputs = tf.matmul(scores, V_)  # (512, 5, 5)*(512, 5, 9)->(512, 5, 9)
    # ---------------------------------如果启用下面的，那么attention_score将作为相应embedding置0的概率----------------------------------------
    min_values = tf.reduce_min(scores, axis=-1)  # [512,5]
    max_values = tf.reduce_max(scores, axis=-1)  # [512,5]
    random_numbers = tf.random.uniform(shape=(512, 21, 21, 11), minval=tf.expand_dims(tf.expand_dims(min_values, -1), -1), maxval=tf.expand_dims(tf.expand_dims(max_values, -1), -1) * 1.5)
    attention_output_mask = tf.cast(random_numbers <= tf.tile(tf.expand_dims(scores, -1), [1, 1, 1, 11]), dtype=tf.float32)  # [512,5,5,9]

    outputs = tf.reduce_sum(tf.tile(tf.expand_dims(V_, 1), (1, 21, 1, 1)) * attention_output_mask, 2)

    outputs += V_  # 这里加上当前的物品embedding，因为之前设置了对角线元素为0，都没有加上去
    # Restore shape多头计算完后
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)  # (512, 5, 9)->[(128,5,9),(128,5,9),(128,5,9),(128,5,9)]->(128,5,36)

    # Residual connection
    outputs += queries  # (128,5,36)
    return outputs


def feedforward(inputs, num_units=[75, 75], dropout_rate=0.5, is_training=True):  # input:[128,5,36]
    # 一维卷积：https://blog.csdn.net/qq_42004289/article/details/105367854
    outputs = tf.layers.conv1d(inputs=inputs, filters=num_units[0], kernel_size=1, activation=tf.nn.relu, use_bias=True)  # input:[128,5,36]，filters=num_units[0]类似全连接层的输出神经元，kernel_size=1就是对序列中的每一项
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))  # input:[128,5,36]
    outputs = tf.layers.conv1d(inputs=outputs, filters=num_units[1], kernel_size=1, activation=tf.nn.relu, use_bias=True)  # input:[128,5,36]
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))  # input:[128,5,36]
    outputs += inputs  # input:[128,5,36]
    return outputs  # output:[128,5,36]


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def layer_normalization(inputs, epsilon=1e-8, scope="ln", reuse=None):  # [128,100,50]
    with tf.variable_scope(scope, reuse=reuse):
        params_shape = inputs.get_shape()[-1:]  # 最后一维的长度，36
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)  # Calculate the mean and variance of `x`，结果形状都为(128, 100, 1)
        gamma = tf.Variable(tf.ones(params_shape))
        beta = tf.Variable(tf.zeros(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def masked_multihead_attention1(inputs, num_units, num_heads, dropout_rate, name="", is_training=True, is_layer_norm=True):
    Q_K_V = tf.layers.dense(inputs, 3 * num_units)
    Q, K, V = tf.split(Q_K_V, 3, -1)
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    print('V_.get_shape()', V_.get_shape().as_list())
    # (h*N, T_q, T_k)
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [hN, T, T]
    align = outputs / (36 ** 0.5)
    # align = general_attention(Q_, K_)
    print('align.get_shape()', align.get_shape().as_list())
    diag_val = tf.ones_like(align[0, :, :])  # [T, T]
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [T, T]
    key_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])
    padding = tf.ones_like(key_masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), padding, align)  # [h*N, T, T]

    outputs = tf.nn.softmax(outputs)
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    outputs = tf.matmul(outputs, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
    # output linear
    outputs = tf.layers.dense(outputs, num_units)
    # drop_out before residual and layernorm
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    # Residual connection
    outputs += inputs  # (N, T_q, C)
    # Normalize
    if is_layer_norm:
        outputs = layer_normalization(outputs, name=name)  # (N, T_q, C)

    return outputs
