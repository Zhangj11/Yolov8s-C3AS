We use mmyolo as the development framework, you can refer to the official website of mmyolo at https://github.com/open-mmlab/mmyolo
-------------------------------------------------------------------------------------------------

configuration file：configs/custom_dataset
Outcome document：work_dirs/

-------------------------------------------------------------------------------------------------
这里给出一些简单的常用代码示例：
Some simple examples of commonly used code are given here:
train：
python tools/train.py "configuration file path"
such as:
python tools/train.py configs/custom_dataset/yolov8_s_200e_drone.py

test:
python tools/test.py "configuration file path" "Training weight paths"
suchu as
python tools/test.py configs/custom_dataset/yolov6_s_syncbn_fast_8xb8_40e_voc.py work_dirs/yolov6_s_syncbn_fast_8xb8_40e_voc/epoch_30.pth 

-------------------------------------------------------------------------------------------------
predict:
python demo/image_demo.py "image path" "configuration file path" "Training weight paths" "Save Path for Prediction Results"
such as:
python demo/image_demo.py demo/2007_007415.jpg ./configs/custom_dataset/yolov6_s_syncbn_fast_8xb8_40e_voc.py ./work_dirs/yolov6_s_syncbn_fast_8xb8_40e_voc/epoch_30.pth --out-dir ./demo/output

-------------------------------------------------------------------------------------------------
FPS测试：
FPS test:
python tools/analysis_tools/benchmark.py "configuration file path" "Training weight paths"
such as
python tools/analysis_tools/benchmark.py configs/custom_dataset/yolov8_s_syncbn_fast_4x8b_30e_voc.py work_dirs/yolov8_s_syncbn_fast_4x8b_30e_voc/best_coco_bbox_mAP_epoch_25.pth

-------------------------------------------------------------------------------------------------
打印模型的 Flops 和 Parameters，并以表格形式展示每层网络复杂度
Print the Flops and Parameters of the model and show the network complexity at each layer in tabular form
python tools/analysis_tools/get_flops.py "configuration file path"