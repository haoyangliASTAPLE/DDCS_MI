if [ ! -d ./.data/umdfaces ];then
    mkdir -p ./.data/umdfaces
else
    echo './.data/umdfaces' dir exist
fi

if [ ! -d ./.data/umdfaces/images ];then
    unzip ./datasets/umdfaces/Umdfaces.zip -d ./.data/umdfaces
fi

python ./datasets/reformat.py --dataset umdfaces