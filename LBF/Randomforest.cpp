#include "Randomforest.h"

using namespace std;

Node::Node(){
	left_child_ = NULL;
	right_child_ = NULL;
	is_leaf_ = false;
	threshold_ = 0.0;
	leaf_identity = -1;
	samples_ = -1;
	thre_changed_ = false;
}

Node::Node(Node* left, Node* right, double thres){
	Node(left, right, thres, false);
}

Node::Node(Node* left, Node* right, double thres, bool leaf){
	left_child_ = left;
	right_child_ = right;
	is_leaf_ = leaf;
	threshold_ = thres;
	//offset_ = cv::Point2f(0, 0);
}

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
	cv::RNG random_generator(cv::getTickCount());
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
	
	// get pixel differents features
	cout << "get pixel differences" << endl;
	cv::Mat_<int> pixel_differences(_local_features_num, augmented_images_index.size()); // 500×26900
	for (int i = 0; i < augmented_images_index.size(); i++)//对每个图片，得到500对像素差特征
	{
		cv::Mat_<double> rotation = rotations[i];
		double scale = scales[i];
		for (int j = 0; j < _local_features_num; j++){//500
			FeatureLocations pos = _local_position[j];
			double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;//对其到当前形状
			double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;//对其到当前形状
			delta_x = scale*delta_x*augmented_bboxes[i].width / 2.0;
			delta_y = scale*delta_y*augmented_bboxes[i].height / 2.0;
			int real_x = delta_x + augmented_current_shapes[i](_landmark_index, 0);//在特征点周围采样的点的实际坐标
			int real_y = delta_y + augmented_current_shapes[i](_landmark_index, 1);
			real_x = std::max(0, std::min(real_x, images[augmented_images_index[i]].cols - 1)); // which cols
			real_y = std::max(0, std::min(real_y, images[augmented_images_index[i]].rows - 1)); // which rows
			int tmp = (int)images[augmented_images_index[i]](real_y, real_x); //real_y at first

			delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
			delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
			delta_x = scale*delta_x*augmented_bboxes[i].width / 2.0;
			delta_y = scale*delta_y*augmented_bboxes[i].height / 2.0;
			real_x = delta_x + augmented_current_shapes[i](_landmark_index, 0);//在特征点周围采样的点的实际坐标
			real_y = delta_y + augmented_current_shapes[i](_landmark_index, 1);
			real_x = std::max(0, std::min(real_x, images[augmented_images_index[i]].cols - 1)); // which cols
			real_y = std::max(0, std::min(real_y, images[augmented_images_index[i]].rows - 1)); // which rows
			pixel_differences(j, i) = tmp - (int)images[augmented_images_index[i]](real_y, real_x);
		}
	}

	// train Random Forest construct each tree in the forest
	//bagging法获得8棵树，构成随机森林
	double overlap = 0.4;
	int step = floor(((double)augmented_images_index.size())*overlap / (_trees_num_per_forest - 1));
	_trees.clear();
	_all_leaf_nodes = 0;
	for (int i = 0; i < _trees_num_per_forest; i++){//8棵树
		int start_index = i*step;
		int end_index = augmented_images_index.size() - (_trees_num_per_forest - i - 1)*step;
		//cv::Mat_<int> data = pixel_differences(cv::Range(0, local_features_num_), cv::Range(start_index, end_index));
		//cv::Mat_<int> sorted_data;
		//cv::sortIdx(data, sorted_data, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
		std::set<int> selected_indexes;//存放阈值特征索引（0-landmark_num）
		std::vector<int> images_indexes;
		for (int j = start_index; j < end_index; j++){
			images_indexes.push_back(j);
		}
		//Node* root = BuildTree(selected_indexes, pixel_differences, images_indexes, 0);
		//trees_.push_back(root);
	}


	return true;
}