# Awesome Tiny Object Detection
A curated list of ```Tiny Object Detection``` papers and related resources.

## Table of Contents

* [Activities](#activities)
* [Papers](#papers)
    * [Tiny Object Detection](#tiny-object-detection)
    * [Tiny Face Detection](#tiny-face-detection)
    * [Tiny Pedestrian Detection](#tiny-pedestrian-detection)
* [Datasets](#datasets)
* [Surveys](#surveys)
* [Articles](#articles)

## Activities

* **Challenge on Small Object Detection for Birds 2023** [[Project]](http://www.mva-org.jp/mva2023/challenge) [[Code]](https://github.com/IIM-TTIJ/MVA2023BirdDetection) 
    * ***MVA 2023***, July 23rd - 25th, 2023, ACT CITY Hamamatsu, Japan 
* **1st Tiny Object Detection (TOD) Challenge Real-world Recognition from Low-quality Inputs (RLQ)** [[Project]](https://rlq-tod.github.io/index.html)
    * ***ECCV 2020***, August 23rd - 28th, 2020, SEC, GLASGOW

## Papers

### Tiny Object Detection

* **Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model** [[Paper]](https://arxiv.org/abs/2208.03987) [[Code]](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)
    * Di Wang, Qiming Zhang, Yufei Xu, Jing Zhang, Bo Du, Dacheng Tao, Liangpei Zhang ***IEEE TGRS***
* **RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection** [[Paper]](https://arxiv.org/abs/2208.08738) [[Code]](https://github.com/Chasel-Tsui/mmdet-rfla)
    * Chang Xu, Jinwang Wang, Wen Yang, Huai Yu, Lei Yu, Gui-Song Xia ***ECCV 2022***
* **Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection** [[Paper]](https://arxiv.org/abs/2202.06934) [[Code]](https://github.com/obss/sahi) [[Benchmark]](https://github.com/fcakyon/small-object-detection-benchmark)
    * Fatih Cagatay Akyon, Sinan Onur Altinuc, Alptekin Temizel ***ICIP 2022***
* **Interactive Multi-Class Tiny-Object Detection** [[Paper]](https://arxiv.org/abs/2203.15266) [[Code]](https://github.com/ChungYi347/Interactive-Multi-Class-Tiny-Object-Detection)
    * Chunggi Lee, Seonwook Park, Heon Song, Jeongun Ryu, Sanghoon Kim, Haejoon Kim, Sérgio Pereira, Donggeun Yoo ***CVPR 2022***
* **QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection** [[Paper]](https://arxiv.org/abs/2103.09136) [[Code]](https://github.com/ChenhongyiYang/QueryDet-PyTorch)
    * Chenhongyi Yang, Zehao Huang, Naiyan Wang ***CVPR 2022***
* **Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges** [[Paper]](https://arxiv.org/abs/2102.12219)
    * Jian Ding, Nan Xue, Gui-Song Xia, Xiang Bai, Wen Yang, Micheal Ying Yang, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, Liangpei Zhang ***TPAMI 2021***
* **MRDet: A Multi-Head Network for Accurate Oriented Object Detection in Aerial Images** [[Paper]](https://arxiv.org/abs/2012.13135)
    * Ran Qin, Qingjie Liu, Guangshuai Gao, Di Huang, Yunhong Wang ***TGRS 2021***
* **Attentional Feature Refinement and Alignment Network for Aircraft Detection in SAR Imagery** [[Paper]](https://arxiv.org/abs/2201.07124)
    * Yan Zhao, Lingjun Zhao, Zhong Liu, Dewen Hu, Gangyao Kuang, Li Liu ***Submitted to TGRS***
* **A Normalized Gaussian Wasserstein Distance for Tiny Object Detection** [[Paper]](https://arxiv.org/abs/2110.13389)
    * Jinwang Wang, Chang Xu, Wen Yang, Lei Yu ***arXiv 2021***
* **Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors** [[Paper]](https://arxiv.org/abs/2008.07043) [[Code]](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection)
    * Jingru Yi, Pengxiang Wu, Bo Liu, Qiaoying Huang, Hui Qu, Dimitris Metaxas ***WACV 2021*** 
* **TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-captured Scenarios** [[Paper]](https://arxiv.org/abs/2108.11539)
    * Xingkui Zhu, Shuchang Lyu, Xu Wang, Qi Zhao ***ICCV Workshop 2021***
* **Oriented Bounding Boxes for Small and Freely Rotated Objects** [[Paper]](https://arxiv.org/abs/2104.11854)
    * Mohsen Zand, Ali Etemad, Michael Greenspan ***TGRS 2021***
* **Learning Calibrated-Guidance for Object Detection in Aerial Images** [[Paper]](https://arxiv.org/abs/2103.11399) [[Code]](https://github.com/WeiZongqi/CG-Net)
    * Dong Liang, Zongqi Wei, Dong Zhang, Qixiang Geng, Liyan Zhang, Han Sun, Huiyu Zhou, Mingqiang Wei, Pan Gao ***arXiv 2021***
* **ReDet: A Rotation-equivariant Detector for Aerial Object Detection** [[Paper]](https://arxiv.org/abs/2103.07733) [[Code]](https://github.com/csuhan/ReDet)
    * Jiaming Han, Jian Ding, Nan Xue, Gui-Song Xia ***CVPR 2021***
* **Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges** [[Paper]](https://arxiv.org/abs/2102.12219) [[Code]](https://github.com/dingjiansw101/AerialDetection)
    * Jian Ding, Nan Xue, Gui-Song Xia, Xiang Bai, Wen Yang, Micheal Ying Yang, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, Liangpei Zhang ***arXiv 2021***
* **Effective Fusion Factor in FPN for Tiny Object Detection** [[Paper]](https://arxiv.org/abs/2011.02298)
    * Yuqi Gong, Xuehui Yu, Yao Ding, Xiaoke Peng, Jian Zhao, Zhenjun Han ***WACV 2021***
* **End-to-End Object Detection with Transformers** [[Paper]](https://arxiv.org/abs/2005.12872) [[Code]](https://github.com/facebookresearch/detr)
    * Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko ***ECCV 2020***
* **Corner Proposal Network for Anchor-free, Two-stage Object Detection** [[Paper]](https://arxiv.org/abs/2007.13816) [[Code]](https://github.com/Duankaiwen/CPNDet)
    * Kaiwen Duan, Lingxi Xie, Honggang Qi, Song Bai, Qingming Huang, Qi Tian ***ECCV 2020***
* **HoughNet: Integrating near and long-range evidence for bottom-up object detection** [[Paper]](https://arxiv.org/abs/2007.02355) [[Code]](https://github.com/nerminsamet/houghnet)
    * Nermin Samet, Samet Hicsonmez, Emre Akbas ***ECCV 2020***
* **EfficientDet: Scalable and Efficient Object Detection** [[Paper]](https://arxiv.org/abs/1911.09070) [[Code]](https://github.com/google/automl/tree/master/efficientdet) [[PyTorch]](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) [[PyTorch]](https://github.com/toandaominh1997/EfficientDet.Pytorch) [[PyTorch]](https://github.com/rwightman/efficientdet-pytorch) [[TensorFlow]](https://github.com/xuannianz/EfficientDet)
    * Mingxing Tan, Ruoming Pang, Quoc V. Le ***CVPR 2020***
* **Efficient Object Detection in Large Images Using Deep Reinforcement Learning** [[Paper]](http://openaccess.thecvf.com/content_WACV_2020/papers/Uzkent_Efficient_Object_Detection_in_Large_Images_Using_Deep_Reinforcement_Learning_WACV_2020_paper.pdf)
    * Burak Uzkent, Christopher Yeh, Stefano Ermon ***WACV 2020***
* **Scale Match for Tiny Person Detection** [[Paper]](https://arxiv.org/abs/1912.10664) [[Benchmark]](https://github.com/ucas-vg/TinyBenchmark)
    * Xuehui Yu, Yuqi Gong, Nan Jiang, Qixiang Ye, Zhenjun Han ***WACV 2020***
* **MultiResolution Attention Extractor for Small Object Detection** [[Paper]](https://arxiv.org/abs/2006.05941v1)
    * Fan Zhang, Licheng Jiao, Lingling Li, Fang Liu, Xu Liu ***arXiv 2020***
* **Intrinsic Relationship Reasoning for Small Object Detection** [[Paper]](https://arxiv.org/abs/2009.00833v1)
    * Kui Fu, Jia Li, Lin Ma, Kai Mu, Yonghong Tian ***arXiv 2020***
* **HRDNet: High-resolution Detection Network for Small Objects** [[Paper]](https://arxiv.org/abs/2006.07607)
    * Ziming Liu, Guangyu Gao, Lin Sun, Zhiyuan Fang ***arXiv 2020***
* **Extended Feature Pyramid Network for Small Object Detection** [[Paper]](https://arxiv.org/abs/2003.07021)
    * Chunfang Deng, Mengmeng Wang, Liang Liu, and Yong Liu ***arXiv 2020***
* **MatrixNets: A New Scale and Aspect Ratio Aware Architecture for Object Detection** [[Paper]](https://arxiv.org/abs/2001.03194) [[Code]](https://github.com/arashwan/matrixnet)
    * Abdullah Rashwan, Rishav Agarwal, Agastya Kalra, Pascal Poupart ***arXiv 2020***
* **Cross-dataset Training for Class Increasing Object Detection** [[Paper]](https://arxiv.org/abs/2001.04621)
    * Yongqiang Yao, Yan Wang, Yu Guo, Jiaojiao Lin, Hongwei Qin, Junjie Yan ***arXiv 2020***
* **TBC-Net: A real-time detector for infrared small target detection using semantic constraint** [[Paper]](https://arxiv.org/abs/2001.05852)
    * Mingxin Zhao, Li Cheng, Xu Yang, Peng Feng, Liyuan Liu, Nanjian Wu ***arXiv 2020***
* **RepPoints V2: Verification Meets Regression for Object Detection** [[Paper]](https://arxiv.org/abs/2007.08508) [[Code]](https://github.com/Scalsol/RepPointsV2)
    * Yihong Chen, Zheng Zhang, Yue Cao, Liwei Wang, Stephen Lin, Han Hu **arXiv 2020**
* **M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network** [[Paper]](https://arxiv.org/abs/1811.04533) [[Code]](https://github.com/qijiezhao/M2Det)
    * Qijie Zhao, Tao Sheng, Yongtao Wang, Zhi Tang, Ying Chen, Ling Cai, Haibin Ling ***AAAI 2019***
* **Better to Follow, Follow to Be Better: Towards Precise Supervision of Feature Super-Resolution for Small Object Detection** [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Noh_Better_to_Follow_Follow_to_Be_Better_Towards_Precise_Supervision_ICCV_2019_paper.pdf) [[Project]](http://vision.snu.ac.kr/project_pages/iccv19_noh/views/)
    * Junhyug Noh, Wonho Bae, Wonhee Lee, Jinhwan Seo, Gunhee Kim ***ICCV 2019***
* **Enriched Feature Guided Refinement Network for Object Detection** [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nie_Enriched_Feature_Guided_Refinement_Network_for_Object_Detection_ICCV_2019_paper.pdf) [[Code]](https://github.com/Ranchentx/EFGRNet)
    * Jing Nie, Rao Muhammad Anwer, Hisham Cholakkal, Fahad Shahbaz Khan, Yanwei Pang, Ling Shao ***ICCV 2019***
* **RepPoints: Point Set Representation for Object Detection** [[Paper]](https://arxiv.org/abs/1904.11490) [[Code]](https://github.com/microsoft/RepPoints)
    * Ze Yang, Shaohui Liu, Han Hu, Liwei Wang, Stephen Lin ***ICCV 2019***
* **Scale-Aware Trident Networks for Object Detection** [[Paper]](https://arxiv.org/abs/1901.01892) [[Code]](https://github.com/TuSimple/simpledet/tree/master/models/tridentnet)
    * Yanghao Li, Yuntao Chen, Naiyan Wang, Zhaoxiang Zhang ***ICCV 2019***
* **SCRDet: Towards More Robust Detection for Small, Cluttered and Rotated Objects** [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_SCRDet_Towards_More_Robust_Detection_for_Small_Cluttered_and_Rotated_ICCV_2019_paper.pdf)
    * Xue Yang, Jirui Yang, Junchi Yan, Yue Zhang, Tengfei Zhang, Zhi Guo, Xian Sun, Kun Fu ***ICCV 2019***
* **Clustered Object Detection in Aerial Images** [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Clustered_Object_Detection_in_Aerial_Images_ICCV_2019_paper.pdf)
    * Fan Yang, Heng Fan, Peng Chu, Erik Blasch, Haibin Ling ***ICCV 2019***
* **The Power of Tiling for Small Object Detection** [[Paper]](https://openaccess.thecvf.com/content_CVPRW_2019/papers/UAVision/Unel_The_Power_of_Tiling_for_Small_Object_Detection_CVPRW_2019_paper.pdf)
    * F. Ozge Unel, Burak O. Ozkalayci, Cevahir Cigla ***CVPR Workshop 2019***
* **Learning Object-Wise Semantic Representation for Detection in Remote Sensing Imagery** [[Paper]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Li_Learning_Object-Wise_Semantic_Representation_for_Detection_in_Remote_Sensing_Imagery_CVPRW_2019_paper.pdf)
    * Chengzheng Li, Chunyan Xu, Zhen Cui, Dan Wang, Zequn Jie, Tong Zhang, Jian Yang ***CVPR Workshop 2019***
* **AugFPN: Improving Multi-scale Feature Learning for Object Detection** [[Paper]](https://arxiv.org/abs/1912.05384)
    * Chaoxu Guo, Bin Fan, Qian Zhang, Shiming Xiang, Chunhong Pan ***CoRR 2019, CVPR2020***
* **R2-CNN: Fast Tiny Object Detection in Large-scale Remote Sensing Images** [[Paper]](https://arxiv.org/abs/1902.06042v2)
    * Jiangmiao Pang, Cong Li, Jianping Shi, Zhihai Xu, Huajun Feng ***TGRS 2019***
* **R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object** [[Paper]](https://arxiv.org/abs/1908.05612) [[Code]](https://github.com/Thinklab-SJTU/R3Det_Tensorflow)
    * Yang, Xue and Liu, Qingqing and Yan, Junchi and Li, Ang and Zhiqiang, Zhang and Gang, Yu ***AAAI 2021***
* **SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization** [[Paper]](https://arxiv.org/abs/1912.05027)
    * Xianzhi Du, Tsung-Yi Lin, Pengchong Jin, Golnaz Ghiasi, Mingxing Tan, Yin Cui, Quoc V. Le, Xiaodan Song ***arXiv 2019***
* **Learning Spatial Fusion for Single-Shot Object Detection** [[Paper]](https://arxiv.org/abs/1911.09516) [[Code]](https://github.com/ruinmessi/ASFF)
    * Songtao Liu, Di Huang, Yunhong Wang ***arXiv 2019***
* **Augmentation for small object detection** [[Paper]](https://arxiv.org/abs/1902.07296) [[Code]](https://github.com/gmayday1997/SmallObjectAugmentation)
    * Mate Kisantal, Zbigniew Wojna, Jakub Murawski, Jacek Naruniec, Kyunghyun Cho ***arXiv 2019***
* **Small Object Detection using Context and Attention** [[Paper]](https://arxiv.org/abs/1912.06319)
    * Jeong-Seon Lim, Marcella Astrid, Hyun-Jin Yoon, Seung-Ik Lee ***arXiv 2019***
* **Single-Shot Refinement Neural Network for Object Detection** [[Paper]](https://arxiv.org/abs/1711.06897) [[Code]](https://github.com/sfzhang15/RefineDet) [[PyTorch]](https://github.com/dd604/refinedet.pytorch)
    * Shifeng Zhang, Longyin Wen, Xiao Bian, Zhen Lei, Stan Z. Li ***CVPR 2018***
* **An Analysis of Scale Invariance in Object Detection - SNIP** [[Paper]](https://arxiv.org/abs/1711.08189)
    * Bharat Singh, Larry S. Davis ***CVPR 2018***
* **Cascade R-CNN Delving into High Quality Object Detection** [[Paper]](https://arxiv.org/abs/1712.00726) [[Code]](https://github.com/zhaoweicai/cascade-rcnn)
    * Zhaowei Cai, Nuno Vasconcelos ***CVPR 2018***
* **Single-Shot Object Detection with Enriched Semantics** [[Paper]](https://arxiv.org/abs/1712.00433)
    * Zhishuai Zhang, Siyuan Qiao, Cihang Xie, Wei Shen, Bo Wang, Alan L. Yuille ***CVPR 2018***
* **Scale-Transferrable Object Detection** [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1376.pdf) [[Code]](https://github.com/arvention/STDN-PyTorch)
    * Peng Zhou, Bingbing Ni, Cong Geng, Jianguo Hu, Yi Xu ***CVPR 2018***
* **Deep Feature Pyramid Reconfiguration for Object Detection** [[Paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tao_Kong_Deep_Feature_Pyramid_ECCV_2018_paper.pdf)
    * Tao Kong, Fuchun Sun, Wenbing Huang, Huaping Liu ***ECCV 2018***
* **DetNet: A Backbone network for Object Detection** [[Paper]](https://arxiv.org/abs/1804.06215) [[Code]](https://github.com/guoruoqian/DetNet_pytorch)
    * Zeming Li, Chao Peng, Gang Yu, Xiangyu Zhang, Yangdong Deng, Jian Sun ***ECCV 2018***
* **SOD-MTGAN: Small Object Detection via Multi-Task Generative Adversarial Network** [[Paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yongqiang_Zhang_SOD-MTGAN_Small_Object_ECCV_2018_paper.pdf)
    * Yancheng Bai, Yongqiang Zhang, Mingli Ding, Bernard Ghanem ***ECCV 2018***
* **SNIPER: Efficient Multi-Scale Training** [[Paper]](https://arxiv.org/abs/1805.09300) [[Code]](https://github.com/MahyarNajibi/SNIPER)
    * Bharat Singh, Mahyar Najibi, Larry S. Davis ***NeurIPS 2018***
* **YOLOv3: An Incremental Improvement** [[Paper]](https://arxiv.org/abs/1804.02767) [[Project]](https://pjreddie.com/darknet/yolo/) [[Code]](https://github.com/ayooshkathuria/pytorch-yolo-v3)
    * Joseph Redmon, Ali Farhadi ***arXiv 2018***
* **You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery** [[Paper]](https://arxiv.org/abs/1805.09512) [[Code]](https://github.com/avanetten/yolt)
    * Adam Van Etten ***arXiv 2018***
* **MSDNN: Multi-Scale Deep Neural Network for Salient Object Detection** [[Paper]](https://arxiv.org/abs/1801.04187)
    * Fen Xiao, Wenzheng Deng, Liangchan Peng, Chunhong Cao, Kai Hu, Xieping Gao ***arXiv 2018***
* **MDSSD: Multi-scale Deconvolutional Single Shot Detector for Small Objects** [[Paper]](https://arxiv.org/abs/1805.07009)
    * Mingliang Xu, Lisha Cui, Pei Lv, Xiaoheng Jiang, Jianwei Niu, Bing Zhou, Meng Wang ***arXiv 2018***
* **Perceptual Generative Adversarial Networks for Small Object Detection** [[Paper]](https://arxiv.org/abs/1706.05274)
    * Jianan Li, Xiaodan Liang, Yunchao Wei, Tingfa Xu, Jiashi Feng, Shuicheng Yan ***CVPR 2017***
* **Feature Pyramid Networks for Object Detection** [[Paper]](https://arxiv.org/abs/1612.03144)
    * Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie ***CVPR 2017***
* **DSSD : Deconvolutional Single Shot Detector** [[Paper]](https://arxiv.org/abs/1701.06659) [[Code]](https://github.com/chengyangfu/caffe/tree/dssd)
    * Cheng-Yang Fu, Wei Liu, Ananth Ranga, Ambrish Tyagi, Alexander C. Berg ***CVPR 2017***
* **Accurate Single Stage Detector Using Recurrent Rolling Convolution** [[Paper]](https://arxiv.org/abs/1704.05776) [[Code]](https://github.com/xiaohaoChen/rrc_detection)
    * Jimmy Ren, Xiaohao Chen, Jianbo Liu, Wenxiu Sun, Jiahao Pang, Qiong Yan, Yu-Wing Tai, Li Xu ***CVPR 2017***
* **Focal Loss for Dense Object Detection** [[Paper]](https://arxiv.org/abs/1708.02002) [[PyTorch]](https://github.com/yhenon/pytorch-retinanet)
    * Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár ***ICCV 2017***
* **Deformable Convolutional Networks** [[Paper]](https://arxiv.org/abs/1703.06211) [[Code]](https://github.com/msracver/Deformable-ConvNets)
    * Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei ***ICCV 2017***
* **Feature-Fused SSD: Fast Detection for Small Objects** [[Paper]](https://arxiv.org/abs/1709.05054) [[Code]](https://github.com/wnzhyee/Feature-Fused-SSD)
    * Guimei Cao, Xuemei Xie, Wenzhe Yang, Quan Liao, Guangming Shi, Jinjian Wu ***ICGIP 2017***
* **FSSD: Feature Fusion Single Shot Multibox Detector** [[Paper]](https://arxiv.org/abs/1712.00960) [[Code]](https://github.com/lzx1413/CAFFE_SSD/tree/fssd)
    * Zuoxin Li, Fuqiang Zhou ***arXiv 2017***
* **Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks** [[Paper]](https://arxiv.org/abs/1512.04143)
    * Sean Bell, C. Lawrence Zitnick, Kavita Bala, Ross Girshick ***CVPR 2016***

### Tiny Face Detection

* **TinaFace: Strong but Simple Baseline for Face Detection** [[Paper]](https://arxiv.org/abs/2011.13183) [[Code]](https://github.com/Media-Smart/vedadet)
    * Yanjia Zhu, Hongxiang Cai, Shuhan Zhang, Chenhao Wang, Yichao Xiong ***arXiv 2020***
* **Robust Face Detection via Learning Small Faces on Hard Images** [[Paper]](http://openaccess.thecvf.com/content_WACV_2020/papers/Zhang_Robust_Face_Detection_via_Learning_Small_Faces_on_Hard_Images_WACV_2020_paper.pdf) [[Code]](https://github.com/bairdzhang/smallhardface)
    * Zhishuai Zhang, Wei Shen, Siyuan Qiao, Yan Wang, Bo Wang, Alan Yuille ***WACV 2020***
* **Finding Tiny Faces in the Wild with Generative Adversarial Network** [[Paper]](https://ivul.kaust.edu.sa/Documents/Publications/2018/Finding%20Tiny%20Faces%20in%20the%20Wild%20with%20Generative%20Adversarial%20Network.pdf)
    * Yancheng Bai, Yongqiang Zhang, Mingli Ding, Bernard Ghanem ***CVPR 2018***
* **Seeing Small Faces from Robust Anchor’s Perspective** [[Paper]](https://arxiv.org/abs/1802.09058)
    * Chenchen Zhu, Ran Tao, Khoa Luu, Marios Savvides ***CVPR 2018***
* **Face-MagNet: Magnifying Feature Maps to Detect Small Faces** [[Paper]](https://arxiv.org/abs/1803.05258)
    * Pouya Samangouei, Mahyar Najibi, Larry Davis, Rama Chellappa ***WACV 2018***
* **Finding Tiny Faces** [[Paper]](https://arxiv.org/abs/1612.04402) [[Project]](http://www.cs.cmu.edu/~peiyunh/tiny/index.html) [[Code]](https://github.com/peiyunh/tiny)
    * Peiyun Hu, Deva Ramanan ***CVPR 2017***
* **S3FD: Single Shot Scale-invariant Face Detector** [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf)
    * Shifeng Zhang Xiangyu Zhu Zhen Lei∗ Hailin Shi Xiaobo Wang Stan Z. Li ***ICCV 2017***
* **Detecting and counting tiny faces** [[Paper]](https://arxiv.org/abs/1801.06504)
    * Alexandre Attia, Sharone Dayan ***arXiv 2018***

### Tiny Pedestrian Detection

* **High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection** [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.pdf) [[Code]](https://github.com/liuwei16/CSP)
    * Wei Liu, ShengCai Liao, Weiqiang Ren, Weidong Hu, Yinan Yu ***CVPR 2019***
* **Feature Selective Anchor-Free Module for Single-Shot Object Detection** [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Feature_Selective_Anchor-Free_Module_for_Single-Shot_Object_Detection_CVPR_2019_paper.pdf) [[PyTorch]](https://github.com/hdjang/Feature-Selective-Anchor-Free-Module-for-Single-Shot-Object-Detection) [[TensorFlow]](https://github.com/xuannianz/FSAF)
    * Chenchen Zhu, Yihui He, Marios Savvides ***CVPR 2019***
* **Seek and You Will Find: A New Optimized Framework for Efficient Detection of Pedestrian** [[Paper]](https://arxiv.org/abs/1912.10241)
    * Sudip Das, Partha Sarathi Mukherjee, Ujjwal Bhattacharya ***arXiv 2019***
* **Small-scale Pedestrian Detection Based on Somatic Topology Localization and Temporal Feature Aggregation** [[Paper]](https://arxiv.org/abs/1807.01438)
    * Tao Song, Leiyu Sun, Di Xie, Haiming Sun, Shiliang Pu ***ECCV 2018***

## Datasets

* **Detection and Tracking Meet Drones Challenge** [[Paper]](https://arxiv.org/abs/2001.06303) [[Project]](http://aiskyeye.com/) [[Code]](https://github.com/VisDrone/VisDrone-Dataset)
    * Pengfei Zhu, Longyin Wen, Dawei Du, Xiao Bian, Heng Fan, Qinghua Hu, Haibin Ling ***TPAMI 2021***
* **Tiny Object Detection in Aerial Images** [[Paper]](https://drive.google.com/file/d/1IiTp7gilwDCGr8QR_H9Covz8aVK7LXiI/view) [[Code]](https://github.com/jwwangchn/AI-TOD)
    * Jinwang Wang, Wen Yang, Haowen Guo, Ruixiang Zhang, Gui-Song Xia ***ICPR 2021***
* **iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images** [[Paper]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.pdf) [[Project]](https://captain-whu.github.io/iSAID/index.html)
    * Syed Waqas Zamir, Aditya Arora, Akshita Gupta, Salman Khan, Guolei Sun, Fahad Shahbaz Khan, Fan Zhu, Ling Shao, Gui-Song Xia, Xiang Bai ***CVPRW 2019***
* **BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos** [[Paper]](http://openaccess.thecvf.com/content_WACV_2020/papers/Bondi_BIRDSAI_A_Dataset_for_Detection_and_Tracking_in_Aerial_Thermal_WACV_2020_paper.pdf) [[Project]](https://sites.google.com/view/elizabethbondi/dataset)
    * Elizabeth Bondi, Raghav Jain, Palash Aggrawal, Saket Anand, Robert Hannaford, Ashish Kapoor, Jim Piavis, Shital Shah, Lucas Joppa, Bistra Dilkina, Milind Tambe ***WACV 2020***
* **TinyPerson Dataset for Tiny Person Detection** [[Paper]](https://arxiv.org/abs/1912.10664) [[Project]](http://vision.ucas.ac.cn/resource.asp)
    * Yu, Xuehui and Gong, Yuqi and Jiang, Nan and Ye, Qixiang and Han, Zhenjun ***WACV 2020***
* **The EuroCity Persons Dataset: A Novel Benchmark for Object Detection** [[Paper]](https://ieeexplore.ieee.org/document/8634919) [[Project]](https://eurocity-dataset.tudelft.nl/eval/overview/home)
    * Braun, Markus and Krebs, Sebastian and Flohr, Fabian B. and Gavrila, Dariu M. ***TPAMI 2019***
* **WiderPerson: A Diverse Dataset for Dense Pedestrian Detection in the Wild** [[Paper]](https://arxiv.org/abs/1909.12118) [[Project]](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/)
    * Shifeng Zhang, Yiliang Xie, Jun Wan, Hansheng Xia, Stan Z. Li, Guodong Guo ***TMM 2019***
* **DOTA: A Large-scale Dataset for Object Detection in Aerial Images** [[Paper]](https://arxiv.org/abs/1711.10398) [[Project]](https://captain-whu.github.io/DOTA/)
    * Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, Liangpei Zhang ***CVPR 2018***
* **NightOwls: A Pedestrians at Night Dataset** [[Paper]](http://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18b/neumann18b.pdf) [[Project]](https://www.nightowls-dataset.org)
    * Lukáš Neumann, Michelle Karg, Shanshan Zhang, Christian Scharfenberger, Eric Piegert, Sarah Mistr, Olga Prokofyeva, Robert Thiel, Andrea Vedaldi, Andrew Zisserman, and Bernt Schiele ***ACCV 2018***
* **DeepScores – A Dataset for Segmentation, Detection and Classification of Tiny Objects** [[Paper]](https://tuggeluk.github.io/papers/preprint_deepscores.pdf) [[Project]](https://tuggeluk.github.io/deepscores/) [[Code]](https://github.com/tuggeluk/DeepScoresExamples)
    * Lukas Tuggener, Ismail Elezi, Jurgen Schmidhuber, Marcello Pelillo, Thilo Stadelmann ***ICPR 2018***
* **Bosch Small Traffic Lights Dataset** [[Paper]](https://ieeexplore.ieee.org/document/7989163) [[Project]](https://hci.iwr.uni-heidelberg.de/node/6132) [[Code]](https://github.com/bosch-ros-pkg/bstld)
    * Karsten Behrendt, Libor Novak, Rami Botros ***ICRA 2017***
* **CityPersons: A Diverse Dataset for Pedestrian Detection** [[Paper]](https://arxiv.org/abs/1702.05693)
    * Shanshan Zhang, Rodrigo Benenson, Bernt Schiele ***arXiv 2017***
* **WIDER FACE: A Face Detection Benchmark** [[Paper]](https://arxiv.org/abs/1511.06523) [[Project]](http://shuoyang1213.me/WIDERFACE/)
    * Shuo Yang, Ping Luo, Chen Change Loy, Xiaoou Tang ***CVPR 2016***
* **Small Object Dataset** [[Paper]](http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr15-densdet.pdf) [[Project]](http://visal.cs.cityu.edu.hk/downloads/smallobjects/)
    * Zheng Ma, Lei Yu, Antoni B. Chan ***CVPR 2015***
* **Caltech Pedestrian Detection Benchmark** [[Paper]](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/CVPR09pedestrians.pdf) [[Paper]](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/PAMI12pedestrians.pdf) [[Project]](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)
    * Piotr Dollár, Christian Wojek, Bernt Schiele, Pietro Perona ***CVPR 2009, TPAMI 2012***
* **Penn-Fudan Database for Pedestrian Detection and Segmentation** [[Paper]](https://www.seas.upenn.edu/~jshi/papers/obj_det_liming_accv07.pdf) [[Project]](https://www.cis.upenn.edu/~jshi/ped_html/)
    * Liming Wang, Jianbo Shi, Gang Song, I-fan Shen ***ACCV 2007***

## Surveys

* **Model Rubik's Cube: Twisting Resolution, Depth and Width for TinyNets** [[Paper]](https://arxiv.org/abs/2010.14819) [[Code]](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/tinynet)
    * Kai Han, Yunhe Wang, Qiulin Zhang, Wei Zhang, Chunjing Xu, Tong Zhang ***NeurIPS 2020***
* **A Survey of Deep Learning-based Object Detection** [[Paper]](https://arxiv.org/abs/1907.09408)
    * Licheng Jiao, Fan Zhang, Fang Liu, Shuyuan Yang, Lingling Li, Zhixi Feng, Rong Qu ***IEEE Access 2019***
* **Recent Advances in Deep Learning for Object Detection** [[Paper]](https://arxiv.org/abs/1908.03673)
    * Xiongwei Wu, Doyen Sahoo, Steven C.H. Hoi ***CoRR 2019***
* **Imbalance Problems in Object Detection: A Review** [[Paper]](https://arxiv.org/abs/1909.00169) [[Project]](https://github.com/kemaloksuz/ObjectDetectionImbalance)
    * Kemal Oksuz, Baris Can Cam, Sinan Kalkan, Emre Akbas ***TPAMI 2020***
* **Object Detection in 20 Years: A Survey** [[Paper]](https://arxiv.org/abs/1905.05055)
    * Zhengxia Zou, Zhenwei Shi, Yuhong Guo, Jieping Ye ***submitted to TPAMI 2019***
* **Speed/accuracy trade-offs for modern convolutional object detectors** [[Paper]](https://arxiv.org/abs/1611.10012)
    * Jonathan Huang, Vivek Rathod, Chen Sun, Menglong Zhu, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Kevin Murphy ***CVPR 2017***

## Articles

* **[Tackling the Small Object Problem in Object Detection](https://blog.roboflow.com/detect-small-objects/) ([Video](https://www.youtube.com/watch?v=WeQcURbHA7U&ab_channel=Roboflow))**
* **[Small objects detection problem - Medium](https://medium.datadriveninvestor.com/small-objects-detection-problem-c5b430996162)**
* **[提升小目标检测的思路 - Zhihu](https://zhuanlan.zhihu.com/p/121666693?utm_source=ZHShareTargetIDMore&utm_medium=social&utm_oi=1108654922240958464)**
* **[How do you do object detection using CNNs on small objects like ping pong balls? - Quora](https://www.quora.com/How-do-you-do-object-detection-using-CNNs-on-small-objects-like-ping-pong-balls)**
* **[深度学习在 small object detection 有什么进展? - Zhihu](https://www.zhihu.com/question/272322209)**
* **[小目标检测问题中“小目标”如何定义？其主要技术难点在哪？有哪些比较好的传统的或深度学习方法？ - Zhihu](https://www.zhihu.com/question/269877902)**
