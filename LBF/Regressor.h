#ifndef REGRESSOR_H
#define REGRESSOR_H

#include "LBF.h"
#include "Randomforest.h"

class Regressor {
public:
	int _stage;

	std::vector<RandomForest> _rd_forests;
	//std::vector<struct model*> linear_model_x_;
	//std::vector<struct model*> linear_model_y_;

	//struct feature_node* tmp_binary_features;

	Regressor(){};
	~Regressor(){};
	Regressor(const Regressor&){};
	std::vector<cv::Mat_<double> > Train(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<int>& augmented_images_index,
		const std::vector<cv::Mat_<double> >& augmented_ground_truth_shapes,
		const std::vector<BoundingBox>& augmented_bboxes,
		const std::vector<cv::Mat_<double> >& augmented_current_shapes,
		const Parameters& params,
		const int stage);

};

class CascadeRegressor {
public:
	Parameters _params;
	std::vector<cv::Mat_<uchar>> _images;
	std::vector<cv::Mat_<double>> _ground_truth_shapes;
	std::vector<BoundingBox> _bboxes;
	std::vector<Regressor> _regressors;

	CascadeRegressor(){};
	void Train(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<cv::Mat_<double> >& ground_truth_shapes,
		const std::vector<BoundingBox>& bboxes,
		const Parameters& params);
	cv::Mat_<double> Predict(const cv::Mat_<uchar>& image, cv::Mat_<double>& current_shape, BoundingBox& bbox);
	void LoadCascadeRegressor(std::string ModelName);
	void SaveCascadeRegressor(std::string ModelName);
};


#endif