from Dice import dice
from rnn import dynamic_rnn
from utils import *

epsilon = 0.000000001


class Model(object):
    def __init__(self, usernum, itemnum, catnum, tivnum, args, use_negsampling=False, use_softmax=True):
        with tf.name_scope('Inputs'):
            self.is_training = tf.placeholder(tf.bool)  # 为使用Dropout,batch_normalization等
            self.use_softmax = use_softmax  # 为了区分FM类模型
            # 以下是行为序列特征
            self.user_id = tf.placeholder(tf.int32, [None, ], name='user_id')
            self.target_item_id = tf.placeholder(tf.int32, [None, ], name='target_item_id')
            self.target_cat_id = tf.placeholder(tf.int32, [None, ], name='target_cat_id')
            self.history_items_id = tf.placeholder(tf.int32, [None, args.max_sequence], name='history_items_id')
            self.history_cats_id = tf.placeholder(tf.int32, [None, args.max_sequence], name='history_cats_id')
            self.history_items_tiv_target = tf.placeholder(tf.int32, shape=(None, args.max_sequence), name='history_tiv_target_id')  # [128,5]历史行为的时间间隔（这是行为物品距离目标物品的时间间隔）
            self.history_items_tiv_neighbor = tf.placeholder(tf.int32, shape=(None, args.max_sequence), name='history_tiv_neighbor_id')  # [128,5]历史行为的时间间隔（这是相邻行为物品之间的时间间隔）
            self.his_item_behaviors_list = tf.placeholder(tf.int32, shape=[None, None, None], name='his_item_behaviors_list')  # [128,5,3]历史物品的行为序列
            self.his_item_behaviors_tiv_list = tf.placeholder(tf.int32, shape=[None, None, None], name='his_item_behaviors_tiv_list')  # [128,5,3]历史物品的行为序列对应的时间间隔
            self.his_user_items_list = tf.placeholder(tf.int32, shape=[None, None, None, None], name='his_user_items_list')  # [128,5,3,3]历史物品的行为序列购买的物品
            self.his_user_cats_list = tf.placeholder(tf.int32, shape=[None, None, None, None], name='his_user_cats_list')  # [128,5,3,3]历史物品的行为序列购买的物品
            # 需要预测的标签：点击还是未点击
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            # 不是填充的行为标记为1，填充的行为标记为0
            self.mask_pad = tf.placeholder(tf.float32, [None, args.max_sequence], name='mask_pad')
            self.mask_aux = tf.placeholder(tf.float32, [None, args.max_sequence], name='mask_aux')
            self.mask_trend_for_history = tf.placeholder(tf.float32, [None, args.max_sequence], name='mask_trend_for_history')
            self.mask_trend_for_target = tf.placeholder(tf.float32, [None, ], name='mask_trend_for_target')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.use_negsampling = use_negsampling
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph')  # [128,5,5]
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')  # [128,5,5]
            # 学习率
            self.lr = tf.placeholder(tf.float64, [])
        with tf.name_scope('Embedding_layer'):
            # ----------------------------------------用户id_embedding----------------------------------------
            self.lookup_table_user = tf.get_variable("user_id_embedding", [usernum, args.user_emb_dim])
            self.user_id_emb = tf.nn.embedding_lookup(self.lookup_table_user, self.user_id)  # [128,36]
            self.his_item_behaviors_emb = tf.nn.embedding_lookup(self.lookup_table_user, self.his_item_behaviors_list)  # [128,5,3,36]
            tf.summary.histogram('user_id_embedding', self.lookup_table_user)
            # ----------------------------------------物品id_embedding----------------------------------------
            self.lookup_table_item = tf.get_variable("item_id_embedding", [itemnum, args.item_emb_dim])
            self.target_item_id_emb = tf.nn.embedding_lookup(self.lookup_table_item, self.target_item_id)  # [128,36]
            self.history_items_id_emb = tf.nn.embedding_lookup(self.lookup_table_item, self.history_items_id)  # [128,5,36]
            self.his_user_items_id_emb = tf.nn.embedding_lookup(self.lookup_table_item, self.his_user_items_list)  # [128,5,3,3,18]
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.lookup_table_item, self.noclk_mid_batch_ph)  # [128,5,5,18]
            tf.summary.histogram('item_id_embedding', self.lookup_table_item)
            # ----------------------------------------物品类别id_embedding----------------------------------------
            self.lookup_table_cat = tf.get_variable("cat_id_embedding", [catnum, args.cat_emb_dim])
            self.target_cat_id_emb = tf.nn.embedding_lookup(self.lookup_table_cat, self.target_cat_id)  # [128,36]
            self.history_cats_id_emb = tf.nn.embedding_lookup(self.lookup_table_cat, self.history_cats_id)  # [128,5,36]
            self.his_user_cats_id_emb = tf.nn.embedding_lookup(self.lookup_table_cat, self.his_user_cats_list)  # [128,5,3,3,18]
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.lookup_table_cat, self.noclk_cat_batch_ph)  # [128,5,5,18]
            tf.summary.histogram('cat_id_embedding', self.lookup_table_cat)
            # ----------------------------------------物品position_id_embedding----------------------------------------
            self.lookup_table_position = tf.get_variable("lookup_table_position", dtype=tf.float32, shape=[args.max_sequence + 1, args.position_emb_dim])
            self.target_position_emb = tf.nn.embedding_lookup(self.lookup_table_position, tf.zeros(shape=tf.shape(self.user_id), dtype=tf.int32))  # 0是留给目标物品的位置编号
            self.history_position_emb = tf.nn.embedding_lookup(self.lookup_table_position, tf.range(1, tf.shape(self.history_items_id)[1] + 1))  # [5,36]
            self.history_position_emb = tf.tile(tf.expand_dims(self.history_position_emb, 0), [tf.shape(self.user_id)[0], 1, 1])  # [128,5,36]
            tf.summary.histogram('position_embedding', self.lookup_table_position)
            # ----------------------------------------历史行为物品距目标物品间隔embedding----------------------------------------
            self.lookup_table_tiv = tf.get_variable('item_tiv_target_embedding', dtype=tf.float32, shape=[tivnum, args.tiv_emb_dim])
            self.target_tiv_emb = tf.nn.embedding_lookup(self.lookup_table_tiv, tf.zeros(shape=tf.shape(self.user_id), dtype=tf.int32))  # 目标物品到自身的时间间隔为0 [128,36] 这里不能直接用args.batch_size，因为最后有可能不足一个batch_size的，形状可能会出错
            self.history_tiv_target_emb = tf.nn.embedding_lookup(self.lookup_table_tiv, self.history_items_tiv_target)  # [128,5,36]
            self.his_item_behaviors_tiv_emb = tf.nn.embedding_lookup(self.lookup_table_tiv, self.his_item_behaviors_tiv_list)  # [128,5,3,4]
            self.user_id_tiv_emb = tf.nn.embedding_lookup(self.lookup_table_tiv, tf.zeros(shape=tf.shape(self.user_id), dtype=tf.int32))#[128,4]
            # ----------------------------------------相邻行为物品间隔embedding----------------------------------------
            self.first_tiv_emb = tf.nn.embedding_lookup(self.lookup_table_tiv, tf.zeros(shape=tf.shape(self.user_id), dtype=tf.int32))  # 第一个物品到自身的时间间隔为0 [128,36] 这里不能直接用args.batch_size，因为最后有可能不足一个batch_size的，形状可能会出错
            self.history_tiv_neighbor_emb = tf.nn.embedding_lookup(self.lookup_table_tiv, self.history_items_tiv_neighbor)  # [128,5,36]
            tf.summary.histogram('item_tiv_target_embedding', self.lookup_table_tiv)  # 观察tiv_embeddings的分布
            # ----------------------------------------用以上embedding合成新的embedding----------------------------------------
            # ----------------------------------------简单地将物品embedding和类别embedding合并----------------------------------------
            self.target_item_emb = tf.concat([self.target_item_id_emb, self.target_cat_id_emb], -1)  # 目标物品完整的embedding
            self.history_items_emb = tf.concat([self.history_items_id_emb, self.history_cats_id_emb], -1)  # 历史物品完整的embedding
            self.history_items_emb_sum = tf.reduce_sum(self.history_items_emb * tf.expand_dims(self.mask_pad, -1), 1)  # 这里要屏蔽掉填充的，也即令填充的embedding为0
            self.his_user_items_emb = tf.concat([self.his_user_items_id_emb, self.his_user_cats_id_emb], -1)  # [128,5,3,3,36]
            if self.use_negsampling:
                self.noclk_history_item_emb = tf.concat([self.noclk_mid_his_batch_embedded[:, :, :2, :], self.noclk_cat_his_batch_embedded[:, :, :2, :]], -1)  # [128,5,2,36]
            self.time_emb = self.history_tiv_neighbor_emb + self.history_position_emb
            # ----------------------------------------将新的物品embedding和位置embedding合并----------------------------------------
            self.target_item_with_position_emb = tf.concat([self.target_item_emb, self.target_position_emb], -1)  # [128,38]
            self.history_item_with_position_emb = tf.concat([self.history_items_emb, self.history_position_emb], -1)  # [128,5,38]
            self.history_item_with_position_emb_sum = tf.reduce_sum(self.history_item_with_position_emb * tf.expand_dims(self.mask_pad, -1), 1)  # 这里要屏蔽掉填充的，也即令填充的embedding为0[128,38]
            if self.use_negsampling:
                self.noclk_item_his_with_position_emb = tf.concat([self.noclk_history_item_emb, tf.tile(tf.expand_dims(self.history_position_emb, 2), (1, 1, 2, 1))], -1)  # [128,5,2,36]
            # ----------------------------------------包含位置embedding和相邻物品的时间间隔(后一个距离前一个的时间间隔，加在后一个的embedding上)embedding合并----------------------------------------
            # self.first_item_with_position_and_neighbor_tiv_emb = tf.concat([self.history_item_with_position_emb[0], self.first_tiv_emb], -1)#[128,40],[128,4]->[128,44]
            # self.target_item_with_position_and_neighbor_tiv_emb = tf.concat([self.target_item_with_position_emb, self.history_tiv_neighbor_emb[:, -1, :]], -1)
            # self.history_items_with_position_and_neighbor_tiv_emb = tf.concat([self.history_item_with_position_emb[:, 1:, :], self.history_tiv_neighbor_emb[:, :-1, :]], -1)
            # self.history_items_with_position_and_neighbor_tiv_emb = tf.concat([tf.expand_dims(self.first_item_with_position_and_neighbor_tiv_emb, 1), self.history_items_with_position_and_neighbor_tiv_emb[:, 1:, :]], 1)
            # self.history_items_with_position_and_neighbor_tiv_emb_sum = tf.reduce_sum(self.history_items_with_position_and_neighbor_tiv_emb, 1)
            # self.all_items_with_position_and_neighbor_tiv_emb = tf.concat([tf.expand_dims(self.first_item_with_position_and_neighbor_tiv_emb, 1), self.history_items_with_position_and_neighbor_tiv_emb], 1)
            # if self.use_negsampling:
            #     self.noclk_item_his_with_position_and_neighbor_tiv_emb = tf.concat([self.noclk_item_his_with_position_emb, tf.tile(tf.expand_dims(self.history_tiv_target_emb, 2), (1, 1, 2, 1))], -1)  #
            # ----------------------------------------包含位置embedding和距目标物品的时间间隔embedding合并----------------------------------------
            self.target_item_with_position_and_target_tiv_emb = tf.concat([self.target_item_with_position_emb, self.target_tiv_emb], -1)  # [128,5,44]
            self.history_items_with_position_and_target_tiv_emb = tf.concat([self.history_item_with_position_emb, self.history_tiv_target_emb], -1)
            self.history_items_with_position_and_target_tiv_emb_sum = tf.reduce_sum(self.history_items_with_position_and_target_tiv_emb * tf.expand_dims(self.mask_pad, -1), 1)
            self.all_items_with_position_and_target_tiv_emb = tf.concat([self.history_items_with_position_and_target_tiv_emb, tf.expand_dims(self.target_item_with_position_and_target_tiv_emb, 1)], 1)
            self.his_item_behaviors_emb_with_tiv = tf.concat([self.his_item_behaviors_emb,self.his_item_behaviors_tiv_emb],-1)#[128,5,3,36],[128,5,3,4]->[128,5,3,40]
            self.user_id_emb_with_tiv = tf.concat([self.user_id_emb,self.user_id_tiv_emb],-1)#[128,36],[128,4]
            if self.use_negsampling:
                self.noclk_item_his_with_position_and_target_tiv_emb = tf.concat([self.noclk_item_his_with_position_emb, tf.tile(tf.expand_dims(self.history_tiv_target_emb, 2), (1, 1, 2, 1))], -1)  #
        # self.target_item_with_tiv_and_position_emb = tf.layers.dropout(self.target_item_with_tiv_and_position_emb, rate=self.dropout_rate, training=self.is_training)
        # self.history_item_with_tiv_and_position_emb = tf.layers.dropout(self.history_item_with_tiv_and_position_emb, rate=self.dropout_rate, training=self.is_training)
        # self.history_item_with_tiv_and_position_emb = tf.layers.dropout(self.history_item_with_tiv_and_position_emb, rate=self.dropout_rate, training=self.is_training)
        # self.w_for_user_dim = tf.get_variable('w_for_user_dim', dtype=tf.float32, shape=[args.user_emb_dim], initializer=tf.random_normal_initializer)  # 学习用户的每一个维度不同的权重
        # 这里历史行为物品与用户的embedding交互
        # self.history_item_for_user = self.history_item_with_position_emb + tf.tile(tf.expand_dims(self.user_id_emb, 1), [1, tf.shape(self.history_items_id)[-1], 1])  # [128,5,36],[128,36]->[128,5,36],[128,5,36]->[128,5,36]
        # self.target_item_for_user = self.target_item_with_position_emb + self.user_id_emb  # [128,36],[128,36]->[128,36]
        self.all_trends = tf.get_variable("all_trends", [args.all_candidate_trends, args.item_emb_dim + args.cat_emb_dim + args.tiv_emb_dim + args.position_emb_dim])  # (100, 44)
        tf.summary.histogram('all_trends', self.all_trends)  # 观察all_trends的分布
        self.saver = tf.train.Saver()

    def dynamic_item_embedding_with_his_user(self, items_emb=None):
        if items_emb is None:
            items_emb = self.history_items_emb
        outputs, scores = dynamic_item_attention_with_his_user(self.user_id_emb_with_tiv, self.his_item_behaviors_emb_with_tiv)
        tf.summary.histogram('dynamic_item_embedding_with_his_user_scores', scores)  # 观察scores的分布是否出现极端化
        return tf.concat([items_emb, outputs, scores], -1)  # [128,5,36],[128,5,36],[128,5,3]->[128,5,75]

    def dynamic_item_embedding_with_his_user_items(self, items_emb=None,his_user_items_embedding = None):
        if items_emb is None:
            items_emb = self.history_items_emb
        if his_user_items_embedding is None:
            self.his_user_items_embedding=self.his_user_items_emb

        outputs, scores = dynamic_item_attention_with_his_user_items(self.history_items_emb, self.his_user_items_embedding)
        tf.summary.histogram('dynamic_item_embedding_with_his_user_items_scores', scores)  # 观察scores的分布是否出现极端化
        return tf.concat([items_emb, outputs, scores], -1)  # [128,5,36],[128,5,36],[128,5,3]->[128,5,75]

    def build_fcn_net(self, fcn_input, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=fcn_input, name='bn1')  # , training=self.is_training
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='fcn_net1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='fcn_net2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2 if self.use_softmax else 1, activation=None, name='fcn_net3')
        return dnn3

    def build_loss(self, fcn_output, other_loss=None, gamma=2.0, alpha=0.25, sample_weight=None):  # focal loss
        with tf.name_scope('Metrics'):
            if self.use_softmax:  # 这时输进来的最后一个维度是2：[0,1]或[1,0]
                self.y_hat = tf.clip_by_value(tf.nn.softmax(fcn_output), clip_value_min=epsilon, clip_value_max=1.)  # [128,2]
                self.loss = -tf.reduce_mean(tf.log(self.y_hat) * self.target_ph, axis=-1)  # [128]
            else:
                self.y_hat = tf.clip_by_value(tf.nn.sigmoid(fcn_output), clip_value_min=epsilon, clip_value_max=1.)  # [128,1]
                self.loss = - tf.reduce_mean(tf.concat([tf.log(self.y_hat) * self.target_ph[:, 0], tf.log(1 - self.y_hat) * self.target_ph[:, 1]], axis=-1))  # [128]
            # 计算focal loss调节因子
            P_t = tf.where(tf.equal(self.target_ph[:, 0], tf.ones_like(self.loss, dtype=tf.float32)), self.y_hat[:, 0], self.y_hat[:, 1])  # 得到预测准确的概率
            alpha_factor = tf.ones(shape=tf.shape(self.loss)) * alpha  # [128]
            alpha_factor = tf.where(tf.equal(self.target_ph[:, 0], tf.ones_like(self.loss, dtype=tf.float32)), alpha_factor, 1 - alpha_factor)  # [128]
            focal_weight = alpha_factor * tf.pow(1 - P_t, gamma)  # [128]
            # 计算focal loss
            fl_loss = focal_weight * self.loss
            if sample_weight is not None:
                fl_loss = fl_loss * (1 + tf.squeeze(sample_weight, -1))
            self.loss = tf.reduce_mean(fl_loss)
            if self.use_negsampling:
                self.loss += self.aux_loss
            if other_loss != None:
                self.loss += other_loss
            tf.summary.scalar('loss', self.loss)
            # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask_pad, mask_aux=None, att_score=None):
        mask_pad = tf.cast(mask_pad, tf.float32)  # [128,4]
        click_input_ = tf.concat([h_states, click_seq], -1)  # [128,4,36],[128,4,36]->[128,4,72]
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)  # [128,4,36],[128,4,36]->[128,4,72]
        click_probability = self.auxiliary_net(click_input_)[:, :, 0]  # [128,4]
        noclick_probability = self.auxiliary_net(noclick_input_)[:, :, 1]  # [128,4]
        click_loss = - tf.log(click_probability) * mask_pad  # [128,4]
        noclick_loss = - tf.log(noclick_probability) * mask_pad  # [128,4]
        if mask_aux is not None:
            mask_aux = tf.cast(mask_aux, tf.float32)  # [128,4]
            click_loss *= mask_aux  # [128,4]
            noclick_loss *= mask_aux  # [128,4]
        if att_score is not None:
            click_loss *= - tf.log(click_probability) * att_score[:, :-1]
            noclick_loss *= - tf.log(noclick_probability) * att_score[:, :-1]
        aux_loss = tf.reduce_mean(click_loss + noclick_loss)
        tf.summary.scalar('aux_loss', aux_loss)
        return aux_loss

    def auxiliary_net(self, inputs):#
        bn1 = tf.layers.batch_normalization(inputs=inputs, name='auxiliary_net_bn1', reuse=tf.AUTO_REUSE)  # , training=self.is_training
        dnn1 = tf.layers.dense(bn1, 100, activation=tf.nn.sigmoid, name='auxiliary_net_f1', reuse=tf.AUTO_REUSE)
        dnn2 = tf.layers.dense(dnn1, 50, activation=tf.nn.sigmoid, name='auxiliary_net_f2', reuse=tf.AUTO_REUSE)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='auxiliary_net_f3', reuse=tf.AUTO_REUSE)
        y_hat = tf.clip_by_value(tf.nn.softmax(dnn3), clip_value_min=epsilon, clip_value_max=1.)
        return y_hat  # [128,4,2]

    def train(self, sess, inps):
        if self.use_negsampling:
            summary, loss, accuracy, aux_loss, _ = sess.run([self.merged, self.loss, self.accuracy, self.aux_loss, self.optimizer],
                                                            feed_dict={self.user_id: inps[0], self.target_item_id: inps[1], self.target_cat_id: inps[2], self.history_items_id: inps[3], self.history_cats_id: inps[4], self.history_items_tiv_target: inps[5], self.history_items_tiv_neighbor: inps[6], self.his_item_behaviors_list: inps[7], self.his_item_behaviors_tiv_list: inps[8], self.mask_pad: inps[9], self.mask_aux: inps[10], self.mask_trend_for_history: inps[11],
                                                                       self.mask_trend_for_target: inps[12], self.seq_len_ph: inps[13], self.target_ph: inps[14], self.lr: inps[15], self.is_training: inps[16], self.his_user_items_list: inps[17], self.his_user_cats_list: inps[18], self.noclk_mid_batch_ph: inps[19], self.noclk_cat_batch_ph: inps[20]})
            return loss, accuracy, aux_loss, summary
        else:
            summary, loss, accuracy, _ = sess.run([self.merged, self.loss, self.accuracy, self.optimizer],
                                                  feed_dict={self.user_id: inps[0], self.target_item_id: inps[1], self.target_cat_id: inps[2], self.history_items_id: inps[3], self.history_cats_id: inps[4], self.history_items_tiv_target: inps[5], self.history_items_tiv_neighbor: inps[6], self.his_item_behaviors_list: inps[7], self.his_item_behaviors_tiv_list: inps[8], self.mask_pad: inps[9], self.mask_aux: inps[10], self.mask_trend_for_history: inps[11], self.mask_trend_for_target: inps[12],
                                                             self.seq_len_ph: inps[13], self.target_ph: inps[14], self.lr: inps[15], self.is_training: inps[16], self.his_user_items_list: inps[17], self.his_user_cats_list: inps[18]})
            return loss, accuracy, 0, summary

    def calculate(self, sess, inps):
        if self.use_negsampling:
            summary, probs, loss, accuracy, aux_loss = sess.run([self.merged, self.y_hat, self.loss, self.accuracy, self.aux_loss],
                                                                feed_dict={self.user_id: inps[0], self.target_item_id: inps[1], self.target_cat_id: inps[2], self.history_items_id: inps[3], self.history_cats_id: inps[4], self.history_items_tiv_target: inps[5], self.history_items_tiv_neighbor: inps[6], self.his_item_behaviors_list: inps[7], self.his_item_behaviors_tiv_list: inps[8], self.mask_pad: inps[9], self.mask_aux: inps[10], self.mask_trend_for_history: inps[11],
                                                                           self.mask_trend_for_target: inps[12], self.seq_len_ph: inps[13], self.target_ph: inps[14], self.is_training: inps[15], self.his_user_items_list: inps[16], self.his_user_cats_list: inps[17], self.noclk_mid_batch_ph: inps[18], self.noclk_cat_batch_ph: inps[19]})
            return probs, loss, accuracy, aux_loss, summary
        else:
            summary, probs, loss, accuracy = sess.run([self.merged, self.y_hat, self.loss, self.accuracy],
                                                      feed_dict={self.user_id: inps[0], self.target_item_id: inps[1], self.target_cat_id: inps[2], self.history_items_id: inps[3], self.history_cats_id: inps[4], self.history_items_tiv_target: inps[5], self.history_items_tiv_neighbor: inps[6], self.his_item_behaviors_list: inps[7], self.his_item_behaviors_tiv_list: inps[8], self.mask_pad: inps[9], self.mask_aux: inps[10], self.mask_trend_for_history: inps[11], self.mask_trend_for_target: inps[12],
                                                                 self.seq_len_ph: inps[13], self.target_ph: inps[14], self.is_training: inps[15], self.his_user_items_list: inps[16], self.his_user_cats_list: inps[17]})
            return probs, loss, accuracy, 0, summary

    def save(self, sess, path):
        self.saver.save(sess, save_path=path)

    def restore(self, sess, path):
        self.saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


def FMLayer(fea_list, output_dim=1):
    fea_list = tf.stack(fea_list, axis=1)  # [[128,36],[128,36],[128,36],[128,36]]->[128,4,36]
    square_of_sum = tf.reduce_sum(fea_list, axis=1, keep_dims=True) ** 2  # [128,1,36]
    sum_of_square = tf.reduce_sum(fea_list ** 2, axis=1, keep_dims=True)  # [128,1,36]
    fm_term = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=-1, keep_dims=False)  # [128,1,36]->[128,1]
    if output_dim == 2:
        fm_term = tf.concat([fm_term, tf.zeros_like(fm_term)], axis=1)
    return fm_term


class Model_FM(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args, use_negsampling=False, use_softmax=False):
        super(Model_FM, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling, use_softmax=use_softmax)
        # ----------------------------------------一次项------------------------------------------
        w_item_var = tf.get_variable("w_item_var", [itemnum, 1], trainable=True)  # 其实对于类别向量来说就是1乘以权重（也就是选取权重）
        w_cate_var = tf.get_variable("w_cate_var", [catnum, 1], trainable=True)
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)  # 选取偏置
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.target_item_id))  # [128,1]目标物品的权重
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.target_cat_id))  # [128,1]目标类别的权重
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.history_items_id), axis=1))  # [128,5,1]->[128,1]历史物品的权重
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.history_cats_id), axis=1))  # [128,5,1]->[128,1]历史类别的权重
        linear = tf.reduce_sum(tf.concat(wx, axis=1), axis=1) + b  # [128,4]->[128,1]
        # ----------------------------------------二次项------------------------------------------
        fea_list = [self.target_item_id_emb, self.target_cat_id_emb, tf.reduce_sum(self.history_items_id_emb, axis=1), tf.reduce_sum(self.history_cats_id_emb, axis=1)]  # [[128,36],[128,36],[128,36],[128,36]]
        logit = linear + FMLayer(fea_list)  # [128,1]+[128,1]->[128,1]

        # self.l2_loss = 2e-5 * tf.add_n([tf.nn.l2_loss(v) for v in [wx, self.item_batch_eb, self.item_his_batch_eb_sum]])
        self.build_loss(logit)


class Model_FFM(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args, use_negsampling=False, use_softmax=False):
        super(Model_FFM, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling, use_softmax)
        w_item_var = tf.get_variable("w_item_var", [itemnum, 1], trainable=True)
        w_cate_var = tf.get_variable("w_cate_var", [catnum, 1], trainable=True)
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.target_item_id))
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.target_cat_id))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.history_items_id), axis=1))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.history_cats_id), axis=1))
        wx = tf.concat(wx, axis=1)
        linear = tf.reduce_sum(wx, axis=1, keep_dims=True) + b

        with tf.name_scope('FFM_embedding'):
            FFM_item_embedding_var = tf.get_variable("FFM_item_embedding_var", [itemnum, 3, EMBEDDING_DIM], trainable=True)
            FFM_cate_embedding_var = tf.get_variable("FFM_cate_embedding_var", [catnum, 3, EMBEDDING_DIM], trainable=True)
            item_emb = tf.nn.embedding_lookup(FFM_item_embedding_var, self.mid_batch_ph)
            item_his_emb = tf.nn.embedding_lookup(FFM_item_embedding_var, self.mid_his_batch_ph)
            item_his_sum = tf.reduce_sum(item_his_emb, axis=1)

            cate_emb = tf.nn.embedding_lookup(FFM_cate_embedding_var, self.cate_batch_ph)
            cate_his_emb = tf.nn.embedding_lookup(FFM_cate_embedding_var, self.cate_his_batch_ph)
            cate_his_sum = tf.reduce_sum(cate_his_emb, axis=1)

        fea_list = [item_emb, item_his_sum, cate_emb, cate_his_sum]
        feas = tf.stack(fea_list, axis=1)
        num = len(fea_list)
        rows, cols = [], []
        for i in range(num - 1):
            for j in range(i + 1, num):
                rows.append([i, j - 1])
                cols.append([j, i])
        p = tf.transpose(tf.gather_nd(tf.transpose(feas, [1, 2, 0, 3]), rows), [1, 0, 2])
        q = tf.transpose(tf.gather_nd(tf.transpose(feas, [1, 2, 0, 3]), cols), [1, 0, 2])
        ffm_term = tf.reduce_sum(p * q, axis=2)
        ffm_term = tf.reduce_sum(ffm_term, axis=1, keep_dims=True)
        logit = linear + ffm_term
        self.build_loss(logit)


class Model_DeepFM(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args, use_negsampling=False, use_softmax=False):
        super(Model_DeepFM, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling, use_softmax)
        w_item_var = tf.get_variable("w_item_var", [itemnum, 1], trainable=True)
        w_cate_var = tf.get_variable("w_cate_var", [catnum, 1], trainable=True)
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.target_item_id))
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.target_cat_id))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.history_items_id), axis=1))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.history_cats_id), axis=1))
        wx = tf.concat(wx, axis=1)
        linear = tf.reduce_sum(wx, axis=1, keep_dims=True) + b
        fea_list = [self.target_item_id_emb, self.target_cat_id_emb, tf.reduce_sum(self.history_items_id_emb, axis=1), tf.reduce_sum(self.history_cats_id_emb, axis=1)]
        fm_term = FMLayer(fea_list)  # [128,1]

        inp = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum], 1)  # [128,36],[128,36],[128,36]->[128,108]
        logit = self.build_fcn_net(inp, use_dice=False)

        logit = tf.layers.dense(tf.concat([logit, fm_term, linear], axis=1), 1, activation=None, name='fm_fc')  # [128,1],[128,1],[128,1]->[128,1]
        # self.l2_loss = 0.01 * tf.add_n([tf.nn.l2_loss(v) for v in [wx, self.item_batch_eb, self.item_his_batch_eb_sum]])
        self.build_loss(logit)


class Model_DeepFFM(Model):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=False):
        super(Model_DeepFFM, self).__init__(n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling, use_softmax=use_softmax)

        w_item_var = tf.get_variable("w_item_var", [n_mid, 1], trainable=True)
        w_cate_var = tf.get_variable("w_cate_var", [n_mid, 1], trainable=True)
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.mid_batch_ph))
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.cate_batch_ph))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.mid_his_batch_ph), axis=1))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.cate_his_batch_ph), axis=1))
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)

        wx = tf.concat(wx, axis=1)
        linear = tf.reduce_sum(wx, axis=1, keep_dims=True) + b

        with tf.name_scope('FFM_embedding'):

            FFM_item_embedding_var = tf.get_variable("FFM_item_embedding_var", [n_mid, 3, EMBEDDING_DIM], trainable=True)
            FFM_cate_embedding_var = tf.get_variable("FFM_cate_embedding_var", [n_cate, 3, EMBEDDING_DIM], trainable=True)
            item_emb = tf.nn.embedding_lookup(FFM_item_embedding_var, self.mid_batch_ph)
            item_his_emb = tf.nn.embedding_lookup(FFM_item_embedding_var, self.mid_his_batch_ph)
            item_his_sum = tf.reduce_sum(item_his_emb, axis=1)

            cate_emb = tf.nn.embedding_lookup(FFM_cate_embedding_var, self.cate_batch_ph)
            cate_his_emb = tf.nn.embedding_lookup(FFM_cate_embedding_var, self.cate_his_batch_ph)
            cate_his_sum = tf.reduce_sum(cate_his_emb, axis=1)

        fea_list = [item_emb, item_his_sum, cate_emb, cate_his_sum]
        feas = tf.stack(fea_list, axis=1)
        num = len(fea_list)
        rows, cols = [], []
        for i in range(num - 1):
            for j in range(i + 1, num):
                rows.append([i, j - 1])
                cols.append([j, i])
        p = tf.transpose(tf.gather_nd(tf.transpose(feas, [1, 2, 0, 3]), rows), [1, 0, 2])
        q = tf.transpose(tf.gather_nd(tf.transpose(feas, [1, 2, 0, 3]), cols), [1, 0, 2])
        ffm_term = tf.reduce_sum(p * q, axis=2)
        ffm_term = tf.reduce_sum(ffm_term, axis=1, keep_dims=True)

        inp = tf.concat([self.uid_batch_embedded, self.item_batch_eb, self.item_his_batch_eb_sum], 1)
        dnn_term = self.build_fcn_net(inp, use_dice=False)

        logit = dnn_term + linear + ffm_term
        self.build_loss(logit)


def ExtremeFMLayer(feas, dim, output_dim=1):
    num = len(feas)
    feas = tf.stack(feas, axis=1)  # batch, field_num, emb_dim
    hidden_nn_layers = []
    field_nums = [num]
    final_len = 0
    hidden_nn_layers.append(feas)
    final_result = []
    cross_layers = [256, 256, 256]

    split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)

    with tf.variable_scope("xfm", initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
        for idx, layer_size in enumerate(cross_layers):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m, shape=[dim, -1, field_nums[0] * field_nums[-1]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            filters = tf.get_variable(name="f_" + str(idx), shape=[1, field_nums[-1] * field_nums[0], layer_size], dtype=tf.float32)

            curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if idx != len(cross_layers) - 1:
                next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                final_len += int(layer_size / 2)
            else:
                direct_connect = curr_out
                next_hidden = 0
                final_len += layer_size
            field_nums.append(int(layer_size / 2))

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1)

        w_nn_output = tf.get_variable(name='w_nn_output', shape=[final_len, 1], dtype=tf.float32)
        b_nn_output = tf.get_variable(name='b_nn_output', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
        xfm_term = tf.matmul(result, w_nn_output) + b_nn_output

        if output_dim == 2:
            xfm_term = tf.concat([xfm_term, tf.zeros_like(xfm_term)], axis=1)
        return xfm_term


# class Model_xDeepFM(Model):
#     def __init__(self, usernum, itemnum, catnum, tivnum, args, use_negsampling=False, use_softmax=False):
#         super(Model_xDeepFM, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling,  use_softmax)
#
#         w_item_var = tf.get_variable("w_item_var", [itemnum, 1], trainable=True)
#         w_cate_var = tf.get_variable("w_cate_var", [catnum, 1], trainable=True)
#         b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)
#         wx = []
#         wx.append(tf.nn.embedding_lookup(w_item_var, self.target_item_id_emb))
#         wx.append(tf.nn.embedding_lookup(w_cate_var, self.target_cat_id_emb))
#         wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.history_items_id_emb), axis=1))
#         wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.history_cats_id_emb), axis=1))
#         wx = tf.concat(wx, axis=1)
#         linear = tf.reduce_sum(wx, axis=1, keep_dims=True) + b
#         inp = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum], 1)
#         mlp_term = self.build_fcn_net(inp, use_dice=False)
#
#         fea_list = [self.target_item_id_emb, self.target_cat_id_emb, tf.reduce_sum(self.history_items_id_emb, axis=1), tf.reduce_sum(self.history_cats_id_emb, axis=1)]
#         fm_term = ExtremeFMLayer(fea_list, args.)
#         self.build_loss(mlp_term + fm_term)
class Model_WideDeep(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(Model_WideDeep, self).__init__(usernum, itemnum, catnum, tivnum, args)
        d_layer_wide = tf.concat([self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum], axis=-1)
        d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')

        fcn_input = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum], 1)
        # Fully connected layer
        bn1 = tf.layers.batch_normalization(inputs=fcn_input, name='bn1')  # , training=self.is_training
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')

        self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


class Model_DNN(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(Model_DNN, self).__init__(usernum, itemnum, catnum, tivnum, args)
        fcn_input = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum], 1)
        self.build_loss(self.build_fcn_net(fcn_input, use_dice=False))


class Model_PNN(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(Model_PNN, self).__init__(usernum, itemnum, catnum, tivnum, args)

        fcn_input = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum], 1)

        # Fully connected layer
        self.build_loss(self.build_fcn_net(fcn_input, use_dice=False))


class Model_DIN(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(Model_DIN, self).__init__(usernum, itemnum, catnum, tivnum, args)
        # Attention layer
        with tf.name_scope('DIN_Attention_layer'):
            self.transposed_history_items_emb = tf.transpose(tf.concat([self.history_items_emb * tf.expand_dims(self.mask_pad, -1), tf.expand_dims(self.target_item_emb, 1)], 1), perm=[0, 2, 1])  # [128,5,36]->[128,36,5]
            self.element_wise_score = element_wise_attention(self.transposed_history_items_emb,self.target_item_emb,self.user_id_emb)  # [128,1,36]
            self.history_items_emb_element_wise_changed = self.element_wise_score * self.history_items_emb
            self.attention_output,_= din_fcn_attention(self.target_item_emb, self.history_items_emb_element_wise_changed, self.mask_pad, din=True, softmax_stag=True)
            self.att_fea = tf.reduce_sum(self.attention_output, axis=1)
            tf.summary.histogram('DIN_Att_fea', self.att_fea)
        fcn_input = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum,self.att_fea], -1)#self.target_item_emb * self.history_items_emb_sum是NFM的体现，这种效果更好一点
            # fcn_input = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.att_fea], -1)#self.target_item_emb * self.att_fea是AFM的体现，这种效果会差一点
        # Fully connected layer
        self.build_loss(self.build_fcn_net(fcn_input, use_dice=True))


class Model_DIN_New(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(Model_DIN_New, self).__init__(usernum, itemnum, catnum, tivnum, args)
        with tf.name_scope('Attention_layer'):
            transposed_history_items_emb = tf.transpose(tf.concat([self.history_items_emb * tf.expand_dims(self.mask_pad, -1), tf.expand_dims(self.target_item_emb, 1)], 1), perm=[0, 2, 1])  # [128,5,36]->[128,36,5]
            element_wise_score = element_wise_attention(transposed_history_items_emb)  # [128,1,36]
            history_items_emb_element_wise_changed = element_wise_score * self.history_items_emb
            dynamic_history_items_emb = self.dynamic_item_embedding_with_his_user_items(history_items_emb_element_wise_changed)
            with tf.variable_scope('item_item_attention'):
                attention_output1, scores = din_fcn_attention(self.target_item_emb, self.history_items_emb, self.mask_pad, din=True, softmax_stag=True)  # SUM:【128,1,36】，List:[128,5,36],scores:[128,5]
            # ---------------------------------如果启用下面的，那么attention_score将作为相应embedding置0的概率----------------------------------------
            min_values = tf.reduce_min(scores, axis=-1)  # [128]
            max_values = tf.reduce_max(scores, axis=-1)  # [128]
            random_numbers = tf.random.uniform(shape=(128, args.max_sequence, args.item_emb_dim + args.cat_emb_dim + args.user_emb_dim + 3), minval=tf.expand_dims(tf.expand_dims(min_values, -1), -1), maxval=tf.expand_dims(tf.expand_dims(max_values, -1), -1) * 1.5)
            attention_output_mask = tf.cast(random_numbers <= tf.tile(tf.expand_dims(scores, -1), [1, 1, tf.shape(dynamic_history_items_emb)[-1]]), dtype=tf.float32)
            masked_attention_output = dynamic_history_items_emb * attention_output_mask
            att_fea1 = tf.reduce_sum(masked_attention_output, 1)  # [128,5,36]->[128,36]  # att_fea2 = tf.reduce_sum(attention_output2, 1)*self.user_id_emb#[128,5,36]->[128,36]
        fcn_input = tf.concat([self.user_id_emb, self.user_id_emb * self.history_items_emb_sum, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum, att_fea1], -1)
        self.build_loss(self.build_fcn_net(fcn_input, use_dice=True))


class Model_DIEN_Gru_att_Gru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIEN_Gru_att_Gru, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.history_items_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, self.mask_pad, softmax_stag=1)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=att_outputs, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum, final_state2], 1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_DIEN_Gru_Gru_att(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIEN_Gru_Gru_att, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.target_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU2_outputs', rnn_outputs2)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs2, ATTENTION_SIZE, self.mask_pad, softmax_stag=1)
            att_fea = tf.reduce_sum(att_outputs, 1)
            tf.summary.histogram('att_fea', att_fea)

        inp = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum, att_fea], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_DIEN_V2_Gru_QA_attGru(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIEN_V2_Gru_QA_attGru, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.history_items_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, self.mask_pad, softmax_stag=1)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(QAAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs, att_scores=tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_DIEN(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(Model_DIEN, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling=True)
        # --------------------------------------------------兴趣提取层--------------------------------------------------
        with tf.name_scope('DIEN_GRU_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(args.rnn_hidden_dim), inputs=self.history_items_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")  # [128,5,24]
            tf.summary.histogram('GRU1_outputs', rnn_outputs)
        # self.aux_loss = self.auxiliary_loss_mask(rnn_outputs[:, :-1, :], self.history_items_emb[:, 1:, :], self.noclk_history_item_emb[:, 1:, :], self.mask_pad[:, 1:], self.mask_aux[:, :-1])
        self.aux_loss = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.history_items_emb[:, 1:, :], self.noclk_history_item_emb[:, 1:, 0, :], self.mask_pad[:, 1:])
        # --------------------------------------------------兴趣演化层--------------------------------------------------
        with tf.name_scope('Attention_layer_1'):
            # 得到的_是[128,100,36],alphas是[128,100]
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, self.mask_pad)  # [128,1,5]
            tf.summary.histogram('alpha_outputs', alphas)
        with tf.name_scope('DIEN_GRU_2'):
            # rnn_outputs2是所有隐状态；final_state2是最后的隐状态
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(args.rnn_hidden_dim), inputs=rnn_outputs, att_scores=tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU2_outputs', rnn_outputs2)
            tf.summary.histogram('GRU2_Final_State', final_state2)
        fcn_input = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum, final_state2], 1)
        self.build_loss(self.build_fcn_net(fcn_input, use_dice=True))


# 这里先计算每个兴趣对于之后所有样本的相关性（注意力机制），然后根据这些分数对之后购买的物品加和，然后计算辅助损失
class Model_DIEN1(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(Model_DIEN1, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling=True)
        # --------------------------------------------------兴趣提取层--------------------------------------------------
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(args.rnn_hidden_dim), inputs=self.history_items_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")  # [128,5,24]
            tf.summary.histogram('GRU_outputs', rnn_outputs)
        # self.aux_loss = self.auxiliary_loss_mask(rnn_outputs[:, :-1, :], self.history_items_emb[:, 1:, :], self.noclk_history_item_emb[:, 1:, :], self.mask_pad[:, 1:], self.mask_aux[:, :-1])
        # 这里先计算每个兴趣对于之后所有样本的相关性（注意力机制），然后根据这些分数对之后购买的物品加和，然后计算辅助损失
        with tf.name_scope('Attention_layer_0'):
            self.history_items_emb1, _ = dien_fcn_attention(rnn_outputs[:, :-1, :], self.history_items_emb[:, 1:, :], self.mask_pad[:, 1:])  # [128,4,36]
            self.history_items_emb2 = tf.reverse(masked_multihead_attention_unite(tf.reverse(self.history_items_emb[:, 1:, :], axis=[1]), tf.reverse(self.history_items_emb[:, 1:, :], axis=[1]), tf.reverse(self.mask_pad[:, 1:], axis=[1])), axis=[1])
        self.aux_loss = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.history_items_emb2, self.noclk_history_item_emb[:, 1:, 0, :], self.mask_pad[:, 1:])
        # --------------------------------------------------兴趣演化层--------------------------------------------------
        with tf.name_scope('Attention_layer_1'):
            # 得到的_是[128,100,36],alphas是[128,100]
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, self.mask_pad)  # [128,36],[128,5,36]->[128,5,36],[128,1,5]
            tf.summary.histogram('alpha_outputs', alphas)
        with tf.name_scope('rnn_2'):
            # rnn_outputs2是所有隐状态；final_state2是最后的隐状态
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(args.rnn_hidden_dim), inputs=rnn_outputs, att_scores=tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU_outputs2', rnn_outputs2)
            tf.summary.histogram('GRU2_Final_State', final_state2)
        fcn_input = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum, final_state2], 1)
        self.build_loss(self.build_fcn_net(fcn_input, use_dice=True))


# 多加一层注意力机制
class Model_DIEN2(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(Model_DIEN2, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling=True)
        # --------------------------------------------------兴趣提取层--------------------------------------------------
        with tf.variable_scope("num_blocks_1"):
            self.user_interest_emb1 = masked_multihead_attention_unite(self.history_items_emb, self.history_items_emb, self.mask_pad, causality=False)  # [128,5,36]
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(args.rnn_hidden_dim), inputs=self.user_interest_emb1, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")  # [128,5,24]
            tf.summary.histogram('GRU_outputs', rnn_outputs)
        # self.aux_loss = self.auxiliary_loss_mask(rnn_outputs[:, :-1, :], self.history_items_emb[:, 1:, :], self.noclk_history_item_emb[:, 1:, :], self.mask_pad[:, 1:], self.mask_aux[:, :-1])

        self.aux_loss = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.history_items_emb[:, 1:, :], self.noclk_history_item_emb[:, 1:, 0, :], self.mask_pad[:, 1:])
        # --------------------------------------------------兴趣演化层--------------------------------------------------
        with tf.name_scope('Attention_layer_1'):
            # 得到的_是[128,100,36],alphas是[128,100]
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, self.mask_pad)  # [128,1,5]
            tf.summary.histogram('alpha_outputs', alphas)
        with tf.name_scope('rnn_2'):
            # rnn_outputs2是所有隐状态；final_state2是最后的隐状态
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(args.rnn_hidden_dim), inputs=rnn_outputs, att_scores=tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU_outputs2', rnn_outputs2)
            tf.summary.histogram('GRU2_Final_State', final_state2)
        fcn_input = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum, final_state2], 1)
        self.build_loss(self.build_fcn_net(fcn_input, use_dice=True))


class Model_BST(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(Model_BST, self).__init__(usernum, itemnum, catnum, tivnum, args)
        self.user_interest_emb = masked_multihead_attention_V2(self.history_items_emb, self.history_items_emb, self.mask_pad)  # [128,5,36]
        self.user_interest_emb = feedforward(self.user_interest_emb, dropout_rate=args.dropout_rate, is_training=self.is_training)  # [128,5,36],
        self.user_interest_emb = tf.reduce_sum(self.user_interest_emb, 1)
        fcn_input = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb, self.target_item_emb * self.history_items_emb_sum, self.user_interest_emb], -1)
        # bn1 = tf.layers.batch_normalization(inputs=fcn_input, name='bn1')  # , training=self.is_training
        # dnn1 = tf.layers.dense(bn1, 1024, activation=None, name='f1')
        # dnn1 = prelu(dnn1, 'prelu1')
        # dnn2 = tf.layers.dense(dnn1, 512, activation=None, name='f2')
        # dnn2 = prelu(dnn2, 'prelu2')
        # dnn3 = tf.layers.dense(dnn2, 256, activation=None, name='f3')
        # dnn3 = prelu(dnn3, 'prelu3')
        # dnn4 = tf.layers.dense(dnn3, 2, activation=None, name='f4')
        self.build_loss(self.build_fcn_net(fcn_input))


class Model_DNN_Multi_Head(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(Model_DNN_Multi_Head, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling=True)
        maxlen = args.max_sequence
        other_embedding_size = 2
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, other_embedding_size])
        self.position_his_emb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
        self.position_his_emb = tf.tile(self.position_his_emb, [tf.shape(self.history_items_emb)[0], 1])  # B*T,E
        self.position_his_emb = tf.reshape(self.position_his_emb, [tf.shape(self.history_items_emb)[0], -1, self.position_his_emb.get_shape().as_list()[1]])  # B,T,E
        with tf.name_scope("multi_head_attention"):
            multihead_attention_outputs = masked_multihead_attention_V2(self.history_items_emb, self.history_items_emb, self.mask_pad)

            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, args.item_emb_dim * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, args.item_emb_dim * 2)
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs  # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
        aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.history_items_emb[:, 1:, :], self.noclk_history_item_emb[:, 1:, :], self.mask_pad[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        inp = tf.concat([self.user_id_emb, self.target_item_emb, self.history_items_emb_sum, self.target_item_emb * self.history_items_emb_sum], 1)
        with tf.name_scope("multi_head_attention"):
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, dropout_rate=0, is_training=True)
            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                print('multihead_attention_outputs_v2.get_shape()', multihead_attention_outputs_v2.get_shape())
                with tf.name_scope('Attention_layer' + str(i)):
                    # 这里使用position embedding来算attention
                    print('self.position_his_emb.get_shape()', self.position_his_emb.get_shape())
                    print('self.item_emb.get_shape()', self.item_emb.get_shape())
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_new(self.item_emb, multihead_attention_outputs_v2, self.position_his_emb, ATTENTION_SIZE, self.mask, stag=str(i))
                    print('attention_output.get_shape()', attention_output.get_shape())
                    att_fea = tf.reduce_sum(attention_output, 1)
                    inp = tf.concat([inp, att_fea], 1)  # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class myModel_V2_old(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(myModel_V2_old, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling=True)
        # -------------------------------增加与目标物品有关的行为序列的权重，类似DIN。能否将它换成简单的内积？-------------------------------
        # with tf.name_scope('Attention_Layer'):
        #     attention_output, scores = din_fcn_attention(self.target_item_with_position_and_target_tiv_emb, self.history_items_with_position_and_target_tiv_emb, self.mask_pad, mode='List', din=True)  # SUM:【128,1,136】，List:[128,5,36];score:[128,5]
        #     history_items_emb_new = tf.expand_dims(self.target_item_with_position_and_target_tiv_emb, 1) * tf.expand_dims(scores, -1) + self.history_items_emb  # [128,5,36]
        #     self.history_items_emb_1, attention_scores = attention(self.target_item_with_position_and_target_tiv_emb, self.history_items_with_position_and_target_tiv_emb, self.mask_pad)  # [128,5,36]
        #     # 将scores(未归一化且mask后的)的和也作为一个特征
        #     self.sum_attention_scores = tf.reshape(tf.reduce_sum(attention_scores, -1), [-1, 1])  # [128,5]->[128,1]
        # -----------------------------------------静态的position_embedding---------------------------------------------------
        # position = positional_encoding(args.max_sequence, args.position_emb_dim)  # [1,50,36]
        # position = tf.tile(position, [tf.shape(self.user_id)[0], 1, 1])
        # position = position[:, :tf.shape(self.history_items_id)[1]]
        # ------------------------------- 根据用户的历史行为序列构建用户的兴趣，这里采用masked-self-attention，模拟DIEN-------------------------------
        self.aux_loss = 0
        with tf.name_scope('masked_self_Attention_Layer'):
            # -------------------------------这里就是单纯地仅用历史行为做self-attention，这里的辅助损失都是以下一个行为为目标-------------------------------
            with tf.variable_scope("num_blocks_1"):
                self.user_interest_emb1 = masked_multihead_attention_unite(self.history_items_with_position_and_target_tiv_emb, self.history_items_with_position_and_target_tiv_emb, self.mask_pad, causality=False)  # [128,5,36]
                self.user_interest_emb1 = feedforward(self.user_interest_emb1, num_units=[88, 44], dropout_rate=args.dropout_rate, is_training=self.is_training)
                self.aux_loss += self.auxiliary_loss(self.user_interest_emb1[:, :-1, :], self.history_items_with_position_and_target_tiv_emb[:, 1:, :], self.noclk_item_his_with_position_and_target_tiv_emb[:, 1:, 0, :], self.mask_pad[:, 1:])
            # -------------------------------这里用上一层的兴趣做self-attention之后再跟目标物品做attention，这里的辅助损失就不能以下一个行为为目标了，只能以目标物品或者干脆不加，
            # -------------------------------由于辅助损失跟attention都是希望能关联起来有关系的，所以这里只用在下面用了attention-------------------------------
            with tf.variable_scope("num_blocks_2"):
                self.user_interest_emb2 = masked_multihead_attention_split(self.user_interest_emb1, self.user_interest_emb1, self.mask_pad, num_units=64, causality=False)  # [4,128,5,9]#并没有将多头合并返回，而是将每个头都分别返回
            user_last_all_interest = []  # 最后的形状是[4,128,9]
            for i, user_interest_emb in enumerate(self.user_interest_emb2):  # [4,128,5,9]->[128,5,9]
                with tf.variable_scope('Multi_interest_for_target_Attention_layer' + str(i)):
                    user_interest_emb_new = feedforward(user_interest_emb, num_units=[32, 16], dropout_rate=args.dropout_rate, is_training=self.is_training)  # [128,5,9]
                    # 这里使用position_embedding和tiv_embedding来算attention[self.history_position_emb, self.history_tiv_target_emb]，每个头使用的attention参数是不一样的
                    user_interest_last, attention_score = attention(self.target_item_with_position_and_target_tiv_emb, [user_interest_emb_new, self.history_tiv_target_emb, self.history_position_emb], self.mask_pad, mode='SUM')
                    user_last_all_interest.append(user_interest_last)  # [4,128,9]
        self.user_last_interest = tf.squeeze(tf.concat(tf.split(user_last_all_interest, 4, axis=0), axis=2), axis=0)  # [128,36]
        # -------------------------------需要注意的是这里是取未填充时最后有效的那个 -------------------------------
        # self.user_interest_emb_last = tf.gather_nd(self.user_interest_emb2, tf.concat([tf.reshape(tf.range(tf.shape(self.seq_len_ph)[0]), [-1, 1]), tf.reshape(self.seq_len_ph - 1, [-1, 1])], -1))  # 取最后一个有效的代表用户的兴趣[128,5,36]->[128,36]
        # self.aux_loss = self.auxiliary_loss_mask(self.user_interest_emb[:, :-1, :], self.history_item_with_position_emb[:, 1:, :], self.noclk_item_his_with_position_emb[:, 1:, :], self.mask_pad[:, 1:], self.mask_aux[:, :-1])
        # ---------------------------------------------------------用户的活跃度---------------------------------------------------------
        # with tf.name_scope('user_activity'):  # dynamic_rnn的第二个返回值是最后一个有效的状态（非padding）
        #     _, self.user_activity = tf.nn.dynamic_rnn(GRUCell(args.rnn_hidden_dim), inputs=self.all_items_with_position_and_target_tiv_emb, sequence_length=self.seq_len_ph, parallel_iterations=51, dtype=tf.float32)
        # ---------------------------------------------------------目标物品的流行度---------------------------------------------------------
        # 这里参考SINE，用户的行为形成的趋势user_trend(z_u)，实际上这个就是注意力分数乘以整个向量，类似于广播
        self.W1 = tf.get_variable("W1", shape=[args.item_emb_dim + args.cat_emb_dim + args.position_emb_dim + args.tiv_emb_dim, args.item_emb_dim + args.cat_emb_dim + args.position_emb_dim + args.tiv_emb_dim])  # (36, 36)
        self.W2 = tf.get_variable("W2", shape=[args.item_emb_dim + args.cat_emb_dim + args.position_emb_dim + args.tiv_emb_dim])  # (36)
        # ----------------------------------以下在计算趋势的时候包括目标物品----------------------------------
        self.all_items_emb = tf.concat([self.history_items_with_position_and_target_tiv_emb, tf.expand_dims(self.target_item_with_position_and_target_tiv_emb, 1)], 1)  # [128,5,36][128,36]->[128,5,36][128,1,36]->[128,6,36]这里的history_items_emb不考虑posiiton
        # 在softmax之前计算mask
        self.mask_trend = tf.concat([self.mask_trend_for_history, tf.expand_dims(self.mask_trend_for_target, 1)], 1)  # [128,5],[128]->[128,5],[128,1]->[128,6]#计算哪些不需要参与当前趋势的计算（mask是根据时间决定），注意这里可以全为0
        self.all_items_emb *= tf.expand_dims(self.mask_trend, -1)  # [128,6,36],[128,6,1]->[128,6,36].注意由于mask_trend可能全为0(但是下面softmax的时候还是会得到相等的分数)，所以这里先乘以mask，这里也可以是全为0，也就是说所有行为（包括目标物品）都不参与当前趋势的计算
        item_list_hidden = tf.math.tanh(tf.einsum('bte,ea->bta', self.all_items_emb, self.W1))  # (128,6,36)
        att_score = tf.einsum('bte,e->bt', item_list_hidden, self.W2)  # [128,6]，我们用einsum求和的话会省略掉reshape的过程，否则矩阵相乘[128,5,36]*[36,1]->[128,5,1]->[128,5]
        # 以下由有填充历史行为的原因，得先mask然后计算softmax得到分数
        mask_paddings = tf.ones_like(self.mask_trend) * (-2 ** 32 + 1)  # [128,6]#这里用mask_trend而不需要mask_pad。因为mask_trend掩盖的实际上大于mask_pad
        att_score = tf.where(tf.equal(self.mask_trend, 0), mask_paddings, att_score)  # [128,6]
        att_score = tf.nn.softmax(att_score)  # 这里有可能全部都要舍掉，但是得出来的分数还都是相等，所以为了避免这种情况，上面先将all_items_emb乘以mask

        user_trend = tf.einsum('bte,bt->be', self.all_items_emb, att_score)  # [128,6,36],[128,6]->[128,36]#用户点击的趋势
        # ----------------------------------用户的一般意图在近期趋势池中激活的趋势----------------------------------
        s_u = tf.einsum('be,ce->bc', user_trend, self.all_trends)  # [128,36],[100,36] (128,100)。用户的趋势在趋势池中的得分
        indices_K = tf.argsort(s_u, axis=-1, direction='DESCENDING')[:, :args.item_trends]  # [128,2]
        c_u = tf.gather(self.all_trends, indices_K)  # [128,2,36]选出当前用户行为有贡献的流行趋势，只有这些趋势才参加反向传播
        #  ----------------------------------为了区别挑选出的趋势，乘以不同的权重----------------------------------
        s_u_k = tf.sort(s_u, axis=-1, direction='DESCENDING')[:, :args.item_trends]  # [128,2]
        c_u = tf.einsum('bke,bk->bke', c_u, tf.math.sigmoid(s_u_k))  # (128,2,36)
        self.item_trend = self.target_item_with_position_and_target_tiv_emb * tf.reduce_sum(c_u, 1)
        self.item_trend_score = tf.reduce_sum(self.target_item_with_position_and_target_tiv_emb * tf.reduce_sum(c_u, 1), axis=-1, keepdims=True)
        # --------------------------------加一个辅助损失，使得趋势池中的趋势尽可能不相似-----------------------------------
        mean_trend = tf.reduce_mean(self.all_trends, axis=1, keepdims=True)
        cov_C = tf.matmul(self.all_trends - mean_trend, tf.transpose(self.all_trends - mean_trend)) / tf.cast(args.item_emb_dim + args.cat_emb_dim, tf.float32)
        F2_C = tf.reduce_sum(tf.math.square(cov_C))
        diag_F2_C = tf.reduce_sum(tf.matrix_diag_part(tf.math.square(cov_C)))
        loss_C = 0.5 * (F2_C - diag_F2_C)
        self.build_loss(self.build_fcn_net(tf.concat([self.user_id_emb, self.history_items_with_position_and_target_tiv_emb_sum, self.target_item_with_position_and_target_tiv_emb, self.history_items_with_position_and_target_tiv_emb_sum * self.target_item_with_position_and_target_tiv_emb, self.user_last_interest, self.item_trend, self.item_trend_score], -1)), loss_C)  # [128,36],[128,36],[128,36],[128,36]


class myModel_V2(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(myModel_V2, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling=True)
        # -------------------------------------------------------动态表示历史物品-------------------------------------------------------
        self.transposed_history_items_emb = tf.transpose(tf.concat([self.history_items_emb * tf.expand_dims(self.mask_pad, -1), tf.expand_dims(self.target_item_emb, 1)], 1), perm=[0, 2, 1])  # [128,5,36]->[128,36,5]
        self.element_wise_score = element_wise_attention(self.transposed_history_items_emb, self.target_item_emb, self.user_id_emb)  # [128,1,36]
        self.history_items_emb_element_wise_changed = self.element_wise_score * self.history_items_emb
        # dynamic_history_items_emb = self.dynamic_item_embedding_with_his_user()  # [128,5,75]
        dynamic_history_items_emb = self.dynamic_item_embedding_with_his_user_items(self.history_items_emb_element_wise_changed)  # [128,5,75]
        # ------------------------------- 根据用户的历史行为序列构建用户的兴趣，这里采用masked-self-attention，模拟DIEN-------------------------------
        self.aux_loss = 0
        with tf.name_scope('masked_self_Attention_Layer'):
            # -------------------------------这里就是单纯地仅用历史行为做self-attention，这里的辅助损失都是以下一个行为为目标-------------------------------
            with tf.variable_scope("num_blocks_1"):
                # 当之前的形状不确定，但是又需要知道的时候，就用reshape确定下来
                # self.user_interest_emb1 = masked_multihead_attention_unite(self.history_items_emb,self.history_items_emb, self.mask_pad, num_units=76, causality=True)  # [128,5,36],[128,5,36],[128,5]
                # self.user_interest_emb1 = masked_multihead_attention_unite(tf.reshape(dynamic_history_items_emb, [args.batch_size, args.max_sequence, args.item_emb_dim + args.cat_emb_dim + args.user_emb_dim + 3 + 4]), tf.reshape(dynamic_history_items_emb, [args.batch_size, args.max_sequence, args.item_emb_dim + args.cat_emb_dim + args.user_emb_dim + 3 +4 ]), self.mask_pad, num_units=76, causality=True)  # [128,5,36],[128,5,36],[128,5]
                self.user_interest_emb1 = masked_multihead_attention_unite(tf.reshape(dynamic_history_items_emb, [args.batch_size, args.max_sequence, 2 * (args.item_emb_dim + args.cat_emb_dim) + 9]), tf.reshape(dynamic_history_items_emb, [args.batch_size, args.max_sequence, 2 * (args.item_emb_dim + args.cat_emb_dim) + 9]), self.mask_pad, num_units=76, causality=True)  # [128,5,36],[128,5,36],[128,5]
                self.user_interest_emb1 = feedforward(self.user_interest_emb1, num_units=[76, 76], dropout_rate=args.dropout_rate, is_training=self.is_
                training)
                self.aux_loss += self.auxiliary_loss(self.user_interest_emb1[:, :-1, :], self.history_items_emb[:, 1:, :], self.noclk_history_item_emb[:, 1:, 0, :], self.mask_pad[:, 1:])
            # -------------------------------这里用上一层的兴趣做self-attention之后再跟目标物品做attention，这里的辅助损失就不能以下一个行为为目标了，只能以目标物品或者干脆不加，
            # -------------------------------由于辅助损失跟attention都是希望能关联起来有关系的，所以这里只用在下面用了attention-------------------------------
            with tf.variable_scope("num_blocks_2"):
                self.user_interest_emb2 = masked_multihead_attention_split(self.user_interest_emb1[:, :-1, :], self.user_interest_emb1[:, :-1, :], self.mask_pad[:, 1:], num_units=64, causality=False)  # [4,128,5,9]#并没有将多头合并返回，而是将每个头都分别返回
            user_last_all_interest = []  # 最后的形状是[4,128,9]
            for i, user_interest_emb in enumerate(self.user_interest_emb2):  # [4,128,5,9]->[128,5,9]
                with tf.variable_scope('Multi_interest_for_target_Attention_layer' + str(i)):
                    user_interest_emb_new = feedforward(user_interest_emb, num_units=[32, 16], dropout_rate=args.dropout_rate, is_training=self.is_training)  # [128,5,9]->[128,5,16]
                    # 这里使用position_embedding和tiv_embedding来算attention[self.history_position_emb, self.history_tiv_target_emb]，每个头使用的attention参数是不一样的
                    user_interest_last, attention_score = attention(self.target_item_emb, user_interest_emb_new, self.mask_pad[:, 1], mode='SUM')  # [128,5,36],[[128,5,9],[128,5,4],[128,5,4]]->[128,9]
                    user_last_all_interest.append(user_interest_last)  # 最后得到[4,128,9] split之后是(1,128,9)

        self.user_last_interest = tf.squeeze(tf.concat(tf.split(user_last_all_interest, 4, axis=0), axis=2), 0)  # [4,128,9] -> [128,36],squeeze的默认参数不能删除，要不然构建图的时候不知道维度
        # ------------------------------- 需要注意的是这里是取未填充时最后有效的那个 -------------------------------
        # self.user_interest_emb_last = tf.gather_nd(self.user_interest_emb2, tf.concat([tf.reshape(tf.range(tf.shape(self.seq_len_ph)[0]), [-1, 1]), tf.reshape(self.seq_len_ph - 1, [-1, 1])], -1))  # 取最后一个有效的代表用户的兴趣[128,5,36]->[128,36]
        # self.aux_loss = self.auxiliary_loss_mask(self.user_interest_emb[:, :-1, :], self.history_item_with_position_emb[:, 1:, :], self.noclk_item_his_with_position_emb[:, 1:, :], self.mask_pad[:, 1:], self.mask_aux[:, :-1])
        # ---------------------------------------------------------用户的活跃度---------------------------------------------------------
        with tf.name_scope('time_decays'):  # dynamic_rnn的第二个返回值是最后一个有效的状态（非padding）
            self.time_decays, _ = tf.nn.dynamic_rnn(GRUCell(args.rnn_hidden_dim), inputs=self.time_emb, sequence_length=self.seq_len_ph, parallel_iterations=51, dtype=tf.float32)

        # ---------------------------------------------------------目标物品的流行度---------------------------------------------------------
        # 这里参考SINE，用户的行为形成的趋势user_trend(z_u)，实际上这个就是注意力分数乘以整个向量，类似于广播
        # self.W1 = tf.get_variable("W1", shape=[args.item_emb_dim + args.cat_emb_dim + args.position_emb_dim + args.tiv_emb_dim, args.item_emb_dim + args.cat_emb_dim + args.position_emb_dim + args.tiv_emb_dim])  # (36, 36)
        # self.W2 = tf.get_variable("W2", shape=[args.item_emb_dim + args.cat_emb_dim + args.position_emb_dim + args.tiv_emb_dim])  # (36)
        #
        # self.history_items_with_position_and_target_tiv_emb *= tf.expand_dims(self.mask_trend_for_history, -1)  # [128,6,36],[128,6,1]->[128,6,36].注意由于mask_trend可能全为0(但是下面softmax的时候还是会得到相等的分数)，所以这里先乘以mask，这里也可以是全为0，也就是说所有行为（包括目标物品）都不参与当前趋势的计算
        # item_list_hidden = tf.math.tanh(tf.einsum('bte,ea->bta', self.history_items_with_position_and_target_tiv_emb, self.W1))  # (128,6,36)
        # att_score = tf.einsum('bte,e->bt', item_list_hidden, self.W2)  # [128,6]，我们用einsum求和的话会省略掉reshape的过程，否则矩阵相乘[128,5,36]*[36,1]->[128,5,1]->[128,5]
        # # 以下由有填充历史行为的原因，得先mask然后计算softmax得到分数
        # mask_paddings = tf.ones_like(self.mask_trend_for_history) * (-2 ** 32 + 1)  # [128,6]#这里用mask_trend而不需要mask_pad。因为mask_trend掩盖的实际上大于mask_pad
        # att_score = tf.where(tf.equal(self.mask_trend_for_history, 0), mask_paddings, att_score)  # [128,6]
        # att_score = tf.nn.softmax(att_score)  # 这里有可能全部都要舍掉，但是得出来的分数还都是相等，所以为了避免这种情况，上面先将all_items_emb乘以mask
        #
        # user_trend = tf.einsum('bte,bt->be', self.history_items_with_position_and_target_tiv_emb, att_score)  # [128,6,36],[128,6]->[128,36]#用户点击的趋势
        # # ----------------------------------用户的一般意图在近期趋势池中激活的趋势----------------------------------
        # s_u = tf.einsum('be,ce->bc', user_trend, self.all_trends)  # [128,36],[100,36] (128,100)。用户的趋势在趋势池中的得分
        # indices_K = tf.argsort(s_u, axis=-1, direction='DESCENDING')[:, :args.item_trends]  # [128,2]
        # c_u = tf.gather(self.all_trends, indices_K)  # [128,2,36]选出当前用户行为有贡献的流行趋势，只有这些趋势才参加反向传播
        # #  ----------------------------------为了区别挑选出的趋势，乘以不同的权重----------------------------------
        # s_u_k = tf.sort(s_u, axis=-1, direction='DESCENDING')[:, :args.item_trends]  # [128,2]
        # c_u = tf.einsum('bke,bk->bke', c_u, tf.math.sigmoid(s_u_k))  # (128,2,36)
        # self.item_trend = tf.reduce_sum(c_u, 1)
        # self.item_trend_score = tf.reduce_sum(self.target_item_with_position_and_target_tiv_emb * tf.reduce_sum(c_u, 1), axis=-1, keepdims=True)
        # # --------------------------------加一个辅助损失，使得趋势池中的趋势尽可能不相似-----------------------------------
        # mean_trend = tf.reduce_mean(self.all_trends, axis=1, keepdims=True)
        # cov_C = tf.matmul(self.all_trends - mean_trend, tf.transpose(self.all_trends - mean_trend)) / tf.cast(args.item_emb_dim + args.cat_emb_dim, tf.float32)
        # F2_C = tf.reduce_sum(tf.math.square(cov_C))
        # diag_F2_C = tf.reduce_sum(tf.matrix_diag_part(tf.math.square(cov_C)))
        # loss_C = 0.5 * (F2_C - diag_F2_C)
        self.build_loss(self.build_fcn_net(tf.concat([self.user_id_emb, self.history_items_emb_sum, self.target_item_emb, self.history_items_emb_sum * self.target_item_emb, self.user_last_interest], -1)),loss_C)  # [128,36],[128,36],[128,36],[128,36]


# 改变了自注意力的计算方式和样本权重的计算
class myModel_V3(Model):
    def __init__(self, usernum, itemnum, catnum, tivnum, args):
        super(myModel_V3, self).__init__(usernum, itemnum, catnum, tivnum, args, use_negsampling=True)
        # ------------------------------- 根据用户的历史行为序列构建用户的兴趣，这里采用masked-se
        # lf-attention，模拟DIEN-------------------------------
        self.aux_loss = 0
        with tf.name_scope('masked_self_Attention_Layer'):
            # -------------------------------这里就是单纯地仅用历史行为做self-attention，这里的辅助损失都是以下一个行为为目标-------------------------------
            with tf.variable_scope("num_blocks_1"):
                self.user_interest_emb1 = masked_multihead_attention_unite(self.all_items_with_position_and_target_tiv_emb, self.all_items_with_position_and_target_tiv_emb, tf.concat([self.mask_pad, tf.ones(shape=[128, 1], dtype=tf.float32)], -1), causality=True)  # [128,5,36],[128,5,36],[128,5+1]
                self.user_interest_emb1 = feedforward(self.user_interest_emb1, num_units=[88, 44], dropout_rate=args.dropout_rate, is_training=self.is_training)
                self.aux_loss += self.auxiliary_loss(self.user_interest_emb1[:, :-2, :], self.history_items_with_position_and_target_tiv_emb[:, 1:, :], self.noclk_item_his_with_position_and_target_tiv_emb[:, 1:, 0, :], self.mask_pad[:, 1:])
            # -------------------------------这里用上一层的兴趣做self-attention之后再跟目标物品做attention，这里的辅助损失就不能以下一个行为为目标了，只能以目标物品或者干脆不加，
            # -------------------------------由于辅助损失跟attention都是希望能关联起来有关系的，所以这里只用在下面用了attention-------------------------------
            with tf.variable_scope("num_blocks_2"):
                self.user_interest_emb2 = masked_multihead_attention_split(self.user_interest_emb1[:, :-1, :], self.user_interest_emb1[:, :-1, :], self.mask_pad, num_units=64, causality=False)  # [4,128,5,9]#并没有将多头合并返回，而是将每个头都分别返回
            user_last_all_interest = []  # 最后的形状是[4,128,9]
            for i, user_interest_emb in enumerate(self.user_interest_emb2):  # [4,128,5,9]->[128,5,9]
                with tf.variable_scope('Multi_interest_for_target_Attention_layer' + str(i)):
                    user_interest_emb_new = feedforward(user_interest_emb, num_units=[32, 16], dropout_rate=args.dropout_rate, is_training=self.is_training)  # [128,5,9]
                    # 这里使用position_embedding和tiv_embedding来算attention[self.history_position_emb, self.history_tiv_target_emb]，每个头使用的attention参数是不一样的
                    user_interest_last, attention_score = attention(self.target_item_with_position_and_target_tiv_emb, [user_interest_emb_new, self.history_tiv_target_emb, self.history_position_emb], self.mask_pad, mode='SUM')  # [128,5,36],[[128,5,9],[128,5,4],[128,5,4]]
                    user_last_all_interest.append(user_interest_last)  # [4,128,9] split之后是(1,128,9)
        self.user_last_interest = tf.squeeze(tf.concat(tf.split(user_last_all_interest, 4, axis=0), axis=2), 0)  # [128,36],squeeze的默认参数不能删除，要不然构建图的时候不知道维度
        # ------------------------------- 需要注意的是这里是取未填充时最后有效的那个 -------------------------------
        # self.user_interest_emb_last = tf.gather_nd(self.user_interest_emb2, tf.concat([tf.reshape(tf.range(tf.shape(self.seq_len_ph)[0]), [-1, 1]), tf.reshape(self.seq_len_ph - 1, [-1, 1])], -1))  # 取最后一个有效的代表用户的兴趣[128,5,36]->[128,36]
        # self.aux_loss = self.auxiliary_loss_mask(self.user_interest_emb[:, :-1, :], self.history_item_with_position_emb[:, 1:, :], self.noclk_item_his_with_position_emb[:, 1:, :], self.mask_pad[:, 1:], self.mask_aux[:, :-1])
        # ---------------------------------------------------------用户的活跃度---------------------------------------------------------
        with tf.name_scope('user_activity'):  # dynamic_rnn的第二个返回值是最后一个有效的状态（非padding）
            self.outputs, self.user_activity = tf.nn.dynamic_rnn(GRUCell(args.rnn_hidden_dim), inputs=self.all_items_with_position_and_target_tiv_emb, sequence_length=self.seq_len_ph, parallel_iterations=51, dtype=tf.float32)
            hidden_1 = tf.layers.dense(self.user_activity, 32, activation=tf.nn.tanh, name='user_activity_fc1')  # [128,80]
            hidden_2 = tf.layers.dense(hidden_1, 16, activation=tf.nn.tanh, name='user_activity_fc2')  # [128,40]
            self.sample_weight = tf.layers.dense(hidden_2, 1, activation=tf.nn.sigmoid, name='user_activity_fc3')  # [128,1]
            tf.summary.histogram('sample_weight', self.sample_weight)
        # ---------------------------------------------------------目标物品的流行度---------------------------------------------------------
        # 这里参考SINE，用户的行为形成的趋势user_trend(z_u)，实际上这个就是注意力分数乘以整个向量，类似于广播
        self.W1 = tf.get_variable("W1", shape=[args.item_emb_dim + args.cat_emb_dim + args.position_emb_dim + args.tiv_emb_dim, args.item_emb_dim + args.cat_emb_dim + args.position_emb_dim + args.tiv_emb_dim])  # (36, 36)
        self.W2 = tf.get_variable("W2", shape=[args.item_emb_dim + args.cat_emb_dim + args.position_emb_dim + args.tiv_emb_dim])  # (36)

        self.history_items_with_position_and_target_tiv_emb *= tf.expand_dims(self.mask_trend_for_history, -1)  # [128,6,36],[128,6,1]->[128,6,36].注意由于mask_trend可能全为0(但是下面softmax的时候还是会得到相等的分数)，所以这里先乘以mask，这里也可以是全为0，也就是说所有行为（包括目标物品）都不参与当前趋势的计算
        item_list_hidden = tf.math.tanh(tf.einsum('bte,ea->bta', self.history_items_with_position_and_target_tiv_emb, self.W1))  # (128,6,36)
        att_score = tf.einsum('bte,e->bt', item_list_hidden, self.W2)  # [128,6]，我们用einsum求和的话会省略掉reshape的过程，否则矩阵相乘[128,5,36]*[36,1]->[128,5,1]->[128,5]
        # 以下由有填充历史行为的原因，得先mask然后计算softmax得到分数
        mask_paddings = tf.ones_like(self.mask_trend_for_history) * (-2 ** 32 + 1)  # [128,6]#这里用mask_trend而不需要mask_pad。因为mask_trend掩盖的实际上大于mask_pad
        att_score = tf.where(tf.equal(self.mask_trend_for_history, 0), mask_paddings, att_score)  # [128,6]
        att_score = tf.nn.softmax(att_score)  # 这里有可能全部都要舍掉，但是得出来的分数还都是相等，所以为了避免这种情况，上面先将all_items_emb乘以mask

        user_trend = tf.einsum('bte,bt->be', self.history_items_with_position_and_target_tiv_emb, att_score)  # [128,6,36],[128,6]->[128,36]#用户点击的趋势
        # ----------------------------------用户的一般意图在近期趋势池中激活的趋势----------------------------------
        s_u = tf.einsum('be,ce->bc', user_trend, self.all_trends)  # [128,36],[100,36] (128,100)。用户的趋势在趋势池中的得分
        indices_K = tf.argsort(s_u, axis=-1, direction='DESCENDING')[:, :args.item_trends]  # [128,2]
        c_u = tf.gather(self.all_trends, indices_K)  # [128,2,36]选出当前用户行为有贡献的流行趋势，只有这些趋势才参加反向传播
        #  ----------------------------------为了区别挑选出的趋势，乘以不同的权重----------------------------------
        s_u_k = tf.sort(s_u, axis=-1, direction='DESCENDING')[:, :args.item_trends]  # [128,2]
        c_u = tf.einsum('bke,bk->bke', c_u, tf.math.sigmoid(s_u_k))  # (128,2,36)
        self.item_trend = tf.reduce_sum(c_u, 1)
        self.item_trend_score = tf.reduce_sum(self.target_item_with_position_and_target_tiv_emb * tf.reduce_sum(c_u, 1), axis=-1, keepdims=True)
        # --------------------------------加一个辅助损失，使得趋势池中的趋势尽可能不相似-----------------------------------
        mean_trend = tf.reduce_mean(self.all_trends, axis=1, keepdims=True)
        cov_C = tf.matmul(self.all_trends - mean_trend, tf.transpose(self.all_trends - mean_trend)) / tf.cast(args.item_emb_dim + args.cat_emb_dim, tf.float32)
        F2_C = tf.reduce_sum(tf.math.square(cov_C))
        diag_F2_C = tf.reduce_sum(tf.matrix_diag_part(tf.math.square(cov_C)))
        loss_C = 0.5 * (F2_C - diag_F2_C)
        self.build_loss(self.build_fcn_net(tf.concat([self.user_id_emb, self.history_items_with_position_and_target_tiv_emb_sum, self.target_item_with_position_and_target_tiv_emb, self.history_items_with_position_and_target_tiv_emb_sum * self.target_item_with_position_and_target_tiv_emb, self.user_last_interest, self.item_trend, self.item_trend_score, self.user_activity, self.sample_weight], -1)), other_loss=loss_C, sample_weight=self.sample_weight)  # [128,36],[128,36],[128,36],[128,36]
