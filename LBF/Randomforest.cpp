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
	cv::Mat_<int> pixel_differences(_local_features_num, augmented_images_index.size()); // 特征数×样本数
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
		set<int> selected_indexes; // 存放阈值特征索引（0-landmark_num）
		vector<int> images_indexes; // 决策树样本子集
		for (int j = start_index; j < end_index; j++){
			images_indexes.push_back(j);
		}

		// train a decision tree
		Node* root = BuildTree(selected_indexes, pixel_differences, images_indexes, 0);
		_trees.push_back(root);
	}

	return true;
}

// 训练一颗决策树
Node* RandomForest::BuildTree(set<int>& selected_indexes, cv::Mat_<int>& pixel_differences,
	vector<int>& images_indexes, int current_depth)
{
	if (images_indexes.size() > 0) // 判断样本集合是否为空
	{
		Node* node = new Node(); // 根节点
		node->depth_ = current_depth;//当前节点深度
		node->samples_ = images_indexes.size();//样本容量
		vector<int> left_indexes, right_indexes; // 左右子节点样本索引

		if (current_depth == _tree_depth) // 判断是否达到最大深度
		{
			node->is_leaf_ = true;
			node->leaf_identity = _all_leaf_nodes;
			return node;
		}

		// 节点分裂
		int ret = FindSplitFeature(node, selected_indexes, pixel_differences, images_indexes, left_indexes, right_indexes);//成功分裂返回0，否则返回1
		
		// actually it won't enter the if block, when the random function is good enough
		if (ret == 1){ // the current node contain all sample when reaches max variance reduction, it is leaf node
			node->is_leaf_ = true;
			node->leaf_identity = _all_leaf_nodes;
			_all_leaf_nodes++;
			return node;
		}

		node->left_child_ = BuildTree(selected_indexes, pixel_differences, left_indexes, current_depth + 1);
		node->right_child_ = BuildTree(selected_indexes, pixel_differences, right_indexes, current_depth + 1);

		return node;
	}
	else
	{
		return NULL;
	}
}

// 节点分裂
int RandomForest::FindSplitFeature(Node* node, std::set<int>& selected_indexes,
	cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int>& left_indexes, std::vector<int>& right_indexes)
{
	//vector<int> val;
	cv::RNG random_generator(cv::getTickCount());
	int threshold;
	double var = -1000000000000.0; // use -DBL_MAX will be better 
	int feature_index = -1; // 最优分裂特征索引值(0 ~ _local_features_num)
	vector<int> tmp_left_indexes, tmp_right_indexes;

	// 从像素差特征中寻找最优分裂阈值: argmax (D - D_L - D_R)
	// 具体操作中会从样本中抽取一个样本在某一序号特征中的像素差值作为分裂阈值，而不是遍历所有样本在所有特征序号处的像素差值
	for (int j = 0; j < _local_features_num; j++)
	{
		if (selected_indexes.find(j) == selected_indexes.end()) //回去看下set ？？？
		{
			tmp_left_indexes.clear();
			tmp_right_indexes.clear();
			double var_lc = 0.0, var_rc = 0.0, var_red = 0.0;//节点数目
			double Ex_2_lc = 0.0, Ex_lc = 0.0, Ey_2_lc = 0.0, Ey_lc = 0.0;//均值的平方与均值
			double Ex_2_rc = 0.0, Ex_rc = 0.0, Ey_2_rc = 0.0, Ey_rc = 0.0;
			// random generate threshold
			std::vector<int> data; // 某一对点映射到样本中的像素差特征
			data.reserve(images_indexes.size());
			for (int i = 0; i < images_indexes.size(); i++){
				data.push_back(pixel_differences(j, images_indexes[i]));
			}
			std::sort(data.begin(), data.end()); // 按大小排序？？？

			int tmp_index = floor((int)(images_indexes.size()*(0.5 + 0.9*(random_generator.uniform(0.0, 1.0) - 0.5)))); //？？？
			int tmp_threshold = data[tmp_index];
			for (int i = 0; i < images_indexes.size(); i++) // 尝试划分
			{
				int index = images_indexes[i];
				if (pixel_differences(j, index) < tmp_threshold) //如果像素差小于阈值
				{
					tmp_left_indexes.push_back(index); // 划分到左子节点
					double value = _regression_targets->at(index)(_landmark_index, 0); // delta x
					Ex_2_lc += pow(value, 2);
					Ex_lc += value;
					value = _regression_targets->at(index)(_landmark_index, 1); // delta y
					Ey_2_lc += pow(value, 2);
					Ey_lc += value;
				}
				else // 划分到右子节点
				{
					tmp_right_indexes.push_back(index);
					double value = _regression_targets->at(index)(_landmark_index, 0);
					Ex_2_rc += pow(value, 2);
					Ex_rc += value;
					value = _regression_targets->at(index)(_landmark_index, 1);
					Ey_2_rc += pow(value, 2);
					Ey_rc += value;
				}
			}

			// 计算方差
			if (tmp_left_indexes.size() == 0) // 如果左子节点为空
			{
				var_lc = 0.0;
			}
			else
			{
				var_lc = Ex_2_lc / tmp_left_indexes.size() - pow(Ex_lc / tmp_left_indexes.size(), 2)
					+ Ey_2_lc / tmp_left_indexes.size() - pow(Ey_lc / tmp_left_indexes.size(), 2);
			}

			if (tmp_right_indexes.size() == 0) // 如果右子节点为空
			{
				var_rc = 0.0;
			}
			else
			{
				var_rc = Ex_2_rc / tmp_right_indexes.size() - pow(Ex_rc / tmp_right_indexes.size(), 2)
					+ Ey_2_rc / tmp_right_indexes.size() - pow(Ey_rc / tmp_right_indexes.size(), 2);
			}
			// 计算方差降低情况
			var_red = -var_lc*tmp_left_indexes.size() - var_rc*tmp_right_indexes.size();

			if (var_red > var)
			{
				var = var_red;
				threshold = tmp_threshold;
				feature_index = j;
				left_indexes = tmp_left_indexes;
				right_indexes = tmp_right_indexes;
			}
		}
	}

	if (feature_index != -1) // actually feature_index will never be -1 
	{
		if (left_indexes.size() == 0 || right_indexes.size() == 0){
			node->is_leaf_ = true; // the node can contain all the samples
			return 1; // 到达叶子节点
		}
		node->threshold_ = threshold; // f分裂阈值
		//node->thre_changed_ = true;
		node->feature_locations_ = _local_position[feature_index]; // 相对索引值
		selected_indexes.insert(feature_index);
		return 0;
	}
	return -1;
}