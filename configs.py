import json
import torch


class Config(object):
    def __init__(self, config, file_path="configs.json"):
        with open(file_path) as config_file:
            self._config = json.load(config_file)
            self._config = self._config.get(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_property(self, property_name):
        return self._config.get(property_name)

    def get_device(self):
        return self.device

    def all(self):
        return self._config


class Create(Config):
    def __init__(self):
        super().__init__('create')

    @property
    def filter_column_value(self):
        return self.get_property('filter_project')

    @property
    def slice_size(self):
        return self.get_property('slice_size')

    @property
    def joern_cli_dir(self):
        return self.get_property('joern_cli_dir')


class Data(Config):
    def __init__(self, config):
        super().__init__(config)

    @property
    def cpg(self):
        return self.get_property('cpg')

    @property
    def raw(self):
        return self.get_property('raw')

    @property
    def input(self):
        return self.get_property('input')

    @property
    def model(self):
        return self.get_property('model')

    @property
    def tokens(self):
        return self.get_property('tokens')

    @property
    def w2v(self):
        return self.get_property('w2v')


class Paths(Data):
    def __init__(self):
        super().__init__('paths')

    @property
    def joern(self):
        return self.get_property('joern')


class Files(Data):
    def __init__(self):
        super().__init__('files')

    @property
    def tokens(self):
        return self.get_property('tokens')

    @property
    def w2v(self):
        return self.get_property('w2v')


# class Embed(Config):
    # def __init__(self):
    #     super().__init__('embed')

    # @property
    # def nodes_dim(self):
    #     return self.get_property('nodes_dim')

    # @property
    # def w2v_args(self):
    #     return self.get_property('word2vec_args')

    # @property
    # def edge_type(self):
    #     return self.get_property('edge_type')


class Embed:
    nodes_dim = 100  # CHANGED from 205
    edge_type = 'Ast'
    
    w2v_args = {
        'vector_size': 100,    # MUST MATCH nodes_dim
        'window': 5,
        'min_count': 3,        # Filter rare words
        'workers': 4,
        'sg': 1,               # Skip-gram
        'hs': 0,
        'negative': 10,
        'epochs': 20,          # More training
        'alpha': 0.025,
        'min_alpha': 0.0001,
        'seed': 42,
    }


class Process(Config):
    def __init__(self):
        super().__init__('process')

    @property
    def epochs(self):
        return self.get_property('epochs')

    @property
    def patience(self):
        return self.get_property('patience')

    @property
    def batch_size(self):
        return self.get_property('batch_size')

    @property
    def dataset_ratio(self):
        return self.get_property('dataset_ratio')

    @property
    def shuffle(self):
        return self.get_property('shuffle')




class Devign:
    learning_rate = 1e-4
    weight_decay = 1.3e-6
    loss_lambda = 1.3e-6
    
    model = {
        'gated_graph_conv_args': {
            'out_channels': 200,
            'num_layers': 8,
            'aggr': 'add',
        },
        'conv_args': {
            'conv1d_1': {
                'in_channels': 100,    # CHANGED from 205 to 100
                'out_channels': 200,
                'kernel_size': 3
            },
            'maxpool1d_1': {
                'kernel_size': 3
            },
            'conv1d_2': {
                'in_channels': 200,
                'out_channels': 200,
                'kernel_size': 1
            },
            'maxpool1d_2': {
                'kernel_size': 2
            },
            'conv1d_3': {
                'in_channels': 200,
                'out_channels': 200,
                'kernel_size': 1
            },
            'maxpool1d_3': {
                'kernel_size': 2
            }
        },
        'concat_size': {
            'dense_concat': 400
        },
        'dense_args': {
            'dense_1': {
                'in_channels': 400,
                'out_channels': 200
            },
            'dense_2': {
                'in_channels': 200,
                'out_channels': 2
            }
        }
    }

# class Devign(Config):
#     def __init__(self):
#         super().__init__('devign')
#         # These values will be loaded from configs.json
#         # Make sure configs.json has these exact values:
#         # "devign": {
#         #     "learning_rate": 1e-4,
#         #     "weight_decay": 1.3e-6,
#         #     "loss_lambda": 1.3e-6,
#         #     "model": {...}
#         # }

#     @property
#     def learning_rate(self):
#         return self.get_property('learning_rate')

#     @property
#     def weight_decay(self):
#         return self.get_property('weight_decay')

#     @property
#     def loss_lambda(self):
#         return self.get_property('loss_lambda')

#     @property
#     def model(self):
#         return self.get_property('model')
