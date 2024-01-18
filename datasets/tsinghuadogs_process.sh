if [ ! -d ./.data ];then
    mkdir -p ./.data
else
    echo './.data' dir exist
fi

unzip ./datasets/tsinghuadogs/low-resolution.zip -d ./.data
mv ./.data/low-resolution ./.data/tsinghuadogs

python ./datasets/reformat.py --dataset tsinghuadogs