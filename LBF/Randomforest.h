#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "LBF.h"

class RandomForest {
public:
	int _stage;
	int _local_features_num;
	int _landmark_index;
	int _tree_depth;
	int _trees_num_per_forest;
	double _local_radius;
	std::vector<cv::Mat_<double> >* _regression_targets;
	std::vector<FeatureLocations> _local_position; // size = param_.local_features_num


	RandomForest(){};
	RandomForest(const Parameters& param, int landmark_index, int stage, std::vector<cv::Mat_<double> >& regression_targets);
	bool TrainForest(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<int>& augmented_images_index,
		const std::vector<BoundingBox>& augmented_bboxes,
		const std::vector<cv::Mat_<double> >& augmented_current_shapes,
		const std::vector<cv::Mat_<double> >& rotations,
		const std::vector<double>& scales);


};

#endif