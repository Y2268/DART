export CXX=g++
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

conf_name='DART'

ROOT_PATH='./experiments/' # The path for saving model and logs

gpus='1,2,'
#gpus='0,'

#P: pretrain SL: soft learning
node_n=1
python -u main_DART.py \
--root-path $ROOT_PATH \
--base 'configs/'$conf_name'.yaml' \
-t True \
--gpus $gpus \
--num-nodes $node_n \

