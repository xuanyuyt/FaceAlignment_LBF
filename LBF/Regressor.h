#ifndef REGRESSOR_H
#define REGRESSOR_H

#include "LBF.h"
#include "Randomforest.h"

class Regressor {
public:
	int _stage;
	Parameters _params;
	std::vector<RandomForest> _rd_forests;
	std::vector<struct model*> _linear_model_x;
	std::vector<struct model*> _linear_model_y;
	std::vector<std::vector<struct model*> > Models_;
	//struct feature_node* tmp_binary_features;
	int leaf_index_count[68];
	Regressor(){
		Models_.resize(global_params._regressor_stages);
	};
	~Regressor(){};
	Regressor(const Regressor&){};
	std::vector<cv::Mat_<double> > Train(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<int>& augmented_images_index,
		const std::vector<cv::Mat_<double> >& augmented_ground_truth_shapes,
		const std::vector<BoundingBox>& augmented_bboxes,
		const std::vector<cv::Mat_<double> >& augmented_current_shapes,
		const Parameters& params,
		const int stage);
	void LoadRegressor(std::ifstream& fin);
	void SaveRegressor(std::ofstream& fout);
	void ConstructLeafCount();
};

class CascadeRegressor {
public:
	Parameters _params;
	std::vector<cv::Mat_<uchar>> _images;
	std::vector<cv::Mat_<double>> _ground_truth_shapes;
	std::vector<BoundingBox> _bboxes;
	std::vector<Regressor> _regressors;
	std::vector<std::vector<struct model*> > _Models;

	CascadeRegressor(){
		_Models.resize(_params._regressor_stages);
	};
	void Train(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<cv::Mat_<double> >& ground_truth_shapes,
		const std::vector<BoundingBox>& bboxes,
		const Parameters& params);
	cv::Mat_<double> Predict(const cv::Mat_<uchar>& image, cv::Mat_<double>& current_shape, BoundingBox& bbox);
	void LoadCascadeRegressor(std::string ModelName);
	void SaveCascadeRegressor(std::string ModelName);
};


#endif