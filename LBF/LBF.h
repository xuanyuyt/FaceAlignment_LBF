#ifndef LBF_H
#define LBF_H

#include <iostream>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include <ctime>

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

// extern 声明，整个工程共享
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

class FeatureLocations
{
public:
	cv::Point2d start;
	cv::Point2d end;
	FeatureLocations(cv::Point2d a, cv::Point2d b){
		start = a;
		end = b;
	}
	FeatureLocations(){
		start = cv::Point2d(0.0, 0.0);
		end = cv::Point2d(0.0, 0.0);
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

// 得到 mean shape, [-1, 1]x[-1, 1]
cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& all_shapes,
	const std::vector<BoundingBox>& all_bboxes);
// project the global shape coordinates to [-1, 1]x[-1, 1]
cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bbox);
// reproject the shape to global coordinates
cv::Mat_<double> ReProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bbox);
// get the rotation and scale parameters by transferring shape_from to shape_to, shape_to = M*shape_from
void SimilarityTransform(const cv::Mat_<double>& shape_to,
	const cv::Mat_<double>& shape_from,
	cv::Mat_<double>& rotation, double& scale);


#endif // !LBF_H
