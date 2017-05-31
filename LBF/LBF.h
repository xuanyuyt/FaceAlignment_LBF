#ifndef LBF_H
#define LBF_H

#include <iostream>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

class Parameters{
public:
	double _bagging_overlap; // 随机森林Bagging参数
	int _trees_num; // 随机森林树的数目
	int _tree_depth; // 每颗树的最大深度
	int _landmarks_num; // landmarks 点数
	int _initial_num; // 多初始形状

	int _regressor_stages; // 回归级数
	std::vector<double> _local_radius; // 局部采样半径
	//std::vector<int> _local_features_num; // 局部采样点数
	int _local_features_num; // 局部采样点数
	cv::Mat_<double> _mean_shape; // 平均形状
};

extern Parameters global_params;
extern std::string modelPath;
extern std::string dataPath;

class BoundingBox{
public:
	double start_x;
	double start_y;
	double width;
	double height;
	double centroid_x;
	double centroid_y;
	BoundingBox(){
		start_x = 0;
		start_y = 0;
		width = 0;
		height = 0;
		centroid_x = 0;
		centroid_y = 0;
	};
};

// 训练数据集
void TrainModel(const char* ModelName, std::vector<std::string> trainDataName);

// 加载常规数据集
void LoadData(std::string file_names,
	std::vector<cv::Mat > &test_images,
	std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<double> >& ground_truth_shapes,
	std::vector<BoundingBox>& bboxes);

// 读取 ground truth shapes
cv::Mat_<double> LoadGroundTruthShape(std::string& filename);

// 将 ground truth shapes 的最小包围矩形作为人脸框
BoundingBox CalculateBoundingBox(cv::Mat_<double>& shape);

#endif // !LBF_H
