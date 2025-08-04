exp_name='DART'

root_path='experiments'
out_root_path='results'

tag='test'
align_test_path='./models/DART/DART/data/test_lfw'
outdir=$out_root_path'/'$exp_name'_'$tag

if [ ! -d $outdir ];then
    mkdir $outdir
fi

python -u scripts/test.py \
--outdir $outdir \
-r ./models/DART/experiments/DART.ckpt \
-c 'configs/DART.yaml' \
--test_path $align_test_path \
--aligned \
--save_features \

