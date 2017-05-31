#include "Randomforest.h"

using namespace std;
using namespace cv;


RandomForest::RandomForest(const Parameters& param, int landmark_index, int stage, std::vector<cv::Mat_<double> >& regression_targets){
	_stage = stage;
	_local_features_num = param._local_features_num;
	_landmark_index = landmark_index;
	_tree_depth = param._tree_depth;
	_trees_num_per_forest = param._trees_num;
	_local_radius = param._local_radius[stage];
	//mean_shape_ = param.mean_shape_;
	_regression_targets = &regression_targets; // get the address pointer, not reference
}

bool RandomForest::TrainForest(
	const std::vector<cv::Mat_<uchar> >& images,
	const std::vector<int>& augmented_images_index,
	const std::vector<BoundingBox>& augmented_bboxes,
	const std::vector<cv::Mat_<double> >& augmented_current_shapes,
	const std::vector<cv::Mat_<double> >& rotations,
	const std::vector<double>& scales)
{
	cout << "build forest of landmark: " << _landmark_index << " of stage: " << _stage << endl;

	// random generate feature locations
	cout << "generate feature locations" << endl;
	RNG random_generator(getTickCount());
	_local_position.clear();//像素差特征初始化
	_local_position.resize(_local_features_num);
	for (int i = 0; i < _local_features_num; i++){//采样500个像素点
		double x, y;
		do{
			x = random_generator.uniform(-_local_radius, _local_radius);
			y = random_generator.uniform(-_local_radius, _local_radius);
		} while (x*x + y*y > _local_radius*_local_radius);
		cv::Point2f a(x, y);

		do{
			x = random_generator.uniform(-_local_radius, _local_radius);
			y = random_generator.uniform(-_local_radius, _local_radius);
		} while (x*x + y*y > _local_radius*_local_radius);
		cv::Point2f b(x, y);

		_local_position[i] = FeatureLocations(a, b);
	}

	return true;
}