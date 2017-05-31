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
	double _bagging_overlap; // ���ɭ��Bagging����
	int _trees_num; // ���ɭ��������Ŀ
	int _tree_depth; // ÿ������������
	int _landmarks_num; // landmarks ����
	int _initial_num; // ���ʼ��״

	int _regressor_stages; // �ع鼶��
	std::vector<double> _local_radius; // �ֲ������뾶
	//std::vector<int> _local_features_num; // �ֲ���������
	int _local_features_num; // �ֲ���������
	cv::Mat_<double> _mean_shape; // ƽ����״
};

// extern �������������̹���
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


// ѵ�����ݼ�
void TrainModel(const char* ModelName, std::vector<std::string> trainDataName);

// ���س������ݼ�
void LoadData(std::string file_names,
	std::vector<cv::Mat > &test_images,
	std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<double> >& ground_truth_shapes,
	std::vector<BoundingBox>& bboxes);

// ��ȡ ground truth shapes
cv::Mat_<double> LoadGroundTruthShape(std::string& filename);

// �� ground truth shapes ����С��Χ������Ϊ������
BoundingBox CalculateBoundingBox(cv::Mat_<double>& shape);

// �õ� mean shape, [-1, 1]x[-1, 1]
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
