需要到https://www.kaggle.com/datasets/zhangtutu123/drone-dataset123/data下载数据集，
Need to go to https://www.kaggle.com/datasets/zhangtutu123/drone-dataset123/data下载数据集.

然后利用下面代码将yolo-style.txt格式的数据集转换成COCO格式：
Then use the following code to convert the dataset in yolo-style.txt format to COCO format：
python tools/dataset_converters/yolo2coco.py /path/to/the/root/dir/of/your_dataset


然后使用下面代码对数据集进行训练集、验证集、以及测试集的划分，如果数据量比较少，可以不划分验证集。下面是划分脚本的具体用法：
Then use the following code to divide the dataset into training set, validation set, and test set, if the amount of data is relatively small, you can not divide the validation set. The following is the specific usage of the division script:

python tools/misc/coco_split.py --json ${COCO label json 路径} \
                                --out-dir ${划分 label json 保存根路径} \
                                --ratios ${划分比例} \
                                [--shuffle] \
                                [--seed ${划分的随机种子}]

在终端输入的时候\可以用一个空格代替
When typing in the terminal \ can be replaced by a space