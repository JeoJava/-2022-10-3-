import pgl
import paddle.fluid.layers as L
import pgl.layers.conv as conv
import paddle.fluid as F
import paddle

class AttentionResGAT(object):
    """Implement of AttentionResGAT"""

    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers")
        self.num_heads = config.get("num_heads")
        self.hidden_size = config.get("hidden_size")
        self.feat_dropout = config.get("feat_drop")
        self.attn_dropout = config.get("attn_drop")
        self.edge_dropout = config.get("edge_dropout")

    def forward(self, graph_wrapper, feature, phase):
        # feature [num_nodes, 100]
        if phase == "train":
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0
        feature = L.fc(feature, size=self.hidden_size * self.num_heads, name="init_feature")
        for i in range(self.num_layers):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout)
            feature = L.batch_norm(feature, name="ln_%s" % i) #使用res+的形式
            feature = L.leaky_relu(feature)

            res_feature = feature
            feature = conv.gat(ngw,
                               feature,
                               self.hidden_size,
                               activation=None,
                               name="gat_layer_%s" % i,
                               num_heads=self.num_heads,
                               feat_drop=self.feat_dropout,
                               attn_drop=self.attn_dropout)
            #print(feature.shape)
            # feature [num_nodes, num_heads * hidden_size]
            feature = res_feature + feature
            attention = L.fc(feature, 1, name="attention") #attention模块
            feature = L.softmax(attention) * feature

            # [num_nodes, num_heads * hidden_size] + [ num_nodes, hidden_size * n_heads]


        ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout)
        feature = conv.gat(ngw,
                           feature,
                           self.num_class,
                           num_heads=1,
                           activation=None,
                           feat_drop=self.feat_dropout,
                           attn_drop=self.attn_dropout,
                           name="output")
        return feature
