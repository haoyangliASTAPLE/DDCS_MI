if [ ! -d ./.data/stanforddogs ];then
    mkdir -p ./.data/stanforddogs
else
    echo './.data/stanforddogs' dir exist
fi

if [ ! -d ./.data/stanforddogs/Images ];then
    tar -cvf ./datasets/stanforddogs/images.tar -C ./.data/stanforddogs
fi

python ./datasets/reformat.py --dataset stanforddogs