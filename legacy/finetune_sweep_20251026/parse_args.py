import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter configuration for training")

    # New argument for GPU index
    parser.add_argument('--gpu', type=str, default=0, help='GPU index to use (e.g. 0, 1, 2, etc.)')

    # Modify to accept a list of values
    parser.add_argument('--n_batch', type=int, nargs='+',           default=256,        help='List of batch sizes (e.g. 256 1024 512)')
    parser.add_argument('--dataset_list', type=str, nargs='+',      default='eatdrinkanother_94u_10',  help='Dataset list')
    parser.add_argument('--finetune_models_list',type=int,nargs='+',default=13,          help='Path to the pre-trained model. Ex: 4, 13, 14, 15, ...')
    parser.add_argument('--model_type_list', type=str, nargs='+',   default='autoencoder_s',    help='Model type list')
    parser.add_argument('--activation', type=str, nargs='+',        default='relu',       help='Activation function for dense layer')
    parser.add_argument('--n_dense', type=int, nargs='+',           default=200,         help='Number of units in the dense layer')
    parser.add_argument('--modee', type=str, nargs='+',             default='classic',     help='Mode of architecture head. Ex: balanced, light, deep')
    parser.add_argument('--learning_rate', type=float, nargs='+',   default=1e-4,           help='Learning rate')

    # Other arguments remain the same
    parser.add_argument('--test_type', type=str,                  default='nusers',       help='Test type')
    parser.add_argument('--k_folds', type=int,                    default=1,              help='Number of folds for cross-validation')
    parser.add_argument('--norm_method_list', type=int,           default=0,              help='Normalization method list')
    parser.add_argument('--n_epochs', type=int,                   default=2,              help='Number of epochs')
    parser.add_argument('--dropout_rate', type=float,             default=0.5,            help='Dropout rate')
    parser.add_argument('--overlap_shift', type=float,            default=0.5,            help='Overlap shift')
    parser.add_argument('--LSTM_layers', type=int,                default=-1,             help='LSTM layers')
    parser.add_argument('--sensors', type=int,                    default=3,              help='Number of sensors')
    parser.add_argument('--seg5', type=bool,                      default=True,           help='Segment 5 flag')

    return parser.parse_args()


# python main.py --model_type_list 'autoencoder_s' --test_type 'nusers' --n_epochs 50 
# --dataset_list 'eatdrinkanother_94u' --learning_rate 1e-5 --dropout_rate 0.5

# python main.py --n_batch 256 1024 512 --model_type_list 'autoencoder_s' --test_type 'nusers' --n_epochs 50 
# --dataset_list 'eatdrinkanother_94u' --learning_rate 1e-5 --dropout_rate 0.5

