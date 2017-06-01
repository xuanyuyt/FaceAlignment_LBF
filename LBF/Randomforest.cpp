#include "Randomforest.h"

using namespace std;

Node::Node(){
	_left_child = NULL;
	_right_child = NULL;
	_is_leaf = false;
	_threshold = 0.0;
	_leaf_identity = -1;
	_samples = -1;
	//_thre_changed = false;
}

Node::Node(Node* left, Node* right, double thres){
	Node(left, right, thres, false);
}

Node::Node(Node* left, Node* right, double thres, bool leaf){
	_left_child = left;
	_right_child = right;
	_is_leaf = leaf;
	_threshold = thres;
	//offset_ = cv::Point2f(0, 0);
}

RandomForest::RandomForest(const Parameters& param, int landmark_index, int stage, std::vector<cv::Mat_<double> >& regression_targets){
	_stage = stage;
	_local_features_num = param._local_features_num;
	_landmark_index = landmark_index;
	_tree_depth = param._tree_depth;
	_trees_num_per_forest = param._trees_num_per_forest;
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
	_local_position.clear();//���ز�������ʼ��
	_local_position.resize(_local_features_num);
	for (int i = 0; i < _local_features_num; i++){//����500�����ص�
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
	cv::Mat_<int> pixel_differences(_local_features_num, augmented_images_index.size()); // ��������������
	for (int i = 0; i < augmented_images_index.size(); i++)//��ÿ��ͼƬ���õ�500�����ز�����
	{
		cv::Mat_<double> rotation = rotations[i];
		double scale = scales[i];
		for (int j = 0; j < _local_features_num; j++){//500
			FeatureLocations pos = _local_position[j];
			double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;//���䵽��ǰ��״
			double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;//���䵽��ǰ��״
			delta_x = scale*delta_x*augmented_bboxes[i].width / 2.0;
			delta_y = scale*delta_y*augmented_bboxes[i].height / 2.0;
			int real_x = delta_x + augmented_current_shapes[i](_landmark_index, 0);//����������Χ�����ĵ��ʵ������
			int real_y = delta_y + augmented_current_shapes[i](_landmark_index, 1);
			real_x = std::max(0, std::min(real_x, images[augmented_images_index[i]].cols - 1)); // which cols
			real_y = std::max(0, std::min(real_y, images[augmented_images_index[i]].rows - 1)); // which rows
			int tmp = (int)images[augmented_images_index[i]](real_y, real_x); //real_y at first

			delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
			delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
			delta_x = scale*delta_x*augmented_bboxes[i].width / 2.0;
			delta_y = scale*delta_y*augmented_bboxes[i].height / 2.0;
			real_x = delta_x + augmented_current_shapes[i](_landmark_index, 0);//����������Χ�����ĵ��ʵ������
			real_y = delta_y + augmented_current_shapes[i](_landmark_index, 1);
			real_x = std::max(0, std::min(real_x, images[augmented_images_index[i]].cols - 1)); // which cols
			real_y = std::max(0, std::min(real_y, images[augmented_images_index[i]].rows - 1)); // which rows
			pixel_differences(j, i) = tmp - (int)images[augmented_images_index[i]](real_y, real_x);
		}
	}

	// train Random Forest construct each tree in the forest
	//bagging�����8�������������ɭ��
	double overlap = 0.4;
	int step = floor(((double)augmented_images_index.size())*overlap / (_trees_num_per_forest - 1));
	_trees.clear();
	_all_leaf_nodes = 0; // Ҷ�ӽڵ���
	for (int i = 0; i < _trees_num_per_forest; i++){//8����
		int start_index = i*step;
		int end_index = augmented_images_index.size() - (_trees_num_per_forest - i - 1)*step;
		set<int> selected_indexes; // �����ֵ����������0-landmark_num��
		vector<int> images_indexes; // �����������Ӽ�
		for (int j = start_index; j < end_index; j++){
			images_indexes.push_back(j);
		}

		// train a decision tree
		Node* root = BuildTree(selected_indexes, pixel_differences, images_indexes, 0);
		_trees.push_back(root);
	}

	return true;
}

// ѵ��һ�ž�����
Node* RandomForest::BuildTree(set<int>& selected_indexes, cv::Mat_<int>& pixel_differences,
	vector<int>& images_indexes, int current_depth)
{
	if (images_indexes.size() > 0) // �ж����������Ƿ�Ϊ��
	{
		Node* node = new Node(); // ���ڵ�
		node->_depth = current_depth;//��ǰ�ڵ����
		node->_samples = images_indexes.size();//��������
		vector<int> left_indexes, right_indexes; // �����ӽڵ���������

		if (current_depth == _tree_depth) // �ж��Ƿ�ﵽ������
		{
			node->_is_leaf = true;
			node->_leaf_identity = _all_leaf_nodes;
			_all_leaf_nodes++;
			return node;
		}

		// �ڵ����
		int ret = FindSplitFeature(node, selected_indexes, pixel_differences, images_indexes, left_indexes, right_indexes);//�ɹ����ѷ���0�����򷵻�1
		
		// actually it won't enter the if block, when the random function is good enough
		if (ret == 1){ // the current node contain all sample when reaches max variance reduction, it is leaf node
			node->_is_leaf = true;
			node->_leaf_identity = _all_leaf_nodes;
			_all_leaf_nodes++;
			return node;
		}

		node->_left_child = BuildTree(selected_indexes, pixel_differences, left_indexes, current_depth + 1);
		node->_right_child = BuildTree(selected_indexes, pixel_differences, right_indexes, current_depth + 1);

		return node;
	}
	else
	{
		return NULL;
	}
}

// �ڵ����
int RandomForest::FindSplitFeature(Node* node, std::set<int>& selected_indexes,
	cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int>& left_indexes, std::vector<int>& right_indexes)
{
	//vector<int> val;
	cv::RNG random_generator(cv::getTickCount());
	int threshold;
	double var = -1000000000000.0; // use -DBL_MAX will be better 
	int feature_index = -1; // ���ŷ�����������ֵ(0 ~ _local_features_num)
	vector<int> tmp_left_indexes, tmp_right_indexes;

	// �����ز�������Ѱ�����ŷ�����ֵ: argmax (D - D_L - D_R)
	// ��������л�������г�ȡһ��������ĳһ��������е����ز�ֵ��Ϊ������ֵ�������Ǳ�����������������������Ŵ������ز�ֵ
	for (int j = 0; j < _local_features_num; j++)
	{
		if (selected_indexes.find(j) == selected_indexes.end()) //��ȥ����set ������
		{
			tmp_left_indexes.clear();
			tmp_right_indexes.clear();
			double var_lc = 0.0, var_rc = 0.0, var_red = 0.0;//�ڵ���Ŀ
			double Ex_2_lc = 0.0, Ex_lc = 0.0, Ey_2_lc = 0.0, Ey_lc = 0.0;//��ֵ��ƽ�����ֵ
			double Ex_2_rc = 0.0, Ex_rc = 0.0, Ey_2_rc = 0.0, Ey_rc = 0.0;
			// random generate threshold
			std::vector<int> data; // ĳһ�Ե�ӳ�䵽�����е����ز�����
			data.reserve(images_indexes.size());
			for (int i = 0; i < images_indexes.size(); i++){
				data.push_back(pixel_differences(j, images_indexes[i]));
			}
			std::sort(data.begin(), data.end()); // ����С���򣿣���

			int tmp_index = floor((int)(images_indexes.size()*(0.5 + 0.9*(random_generator.uniform(0.0, 1.0) - 0.5)))); //������
			int tmp_threshold = data[tmp_index];
			for (int i = 0; i < images_indexes.size(); i++) // ���Ի���
			{
				int index = images_indexes[i];
				if (pixel_differences(j, index) < tmp_threshold) //������ز�С����ֵ
				{
					tmp_left_indexes.push_back(index); // ���ֵ����ӽڵ�
					double value = _regression_targets->at(index)(_landmark_index, 0); // delta x
					Ex_2_lc += pow(value, 2);
					Ex_lc += value;
					value = _regression_targets->at(index)(_landmark_index, 1); // delta y
					Ey_2_lc += pow(value, 2);
					Ey_lc += value;
				}
				else // ���ֵ����ӽڵ�
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

			// ���㷽��
			if (tmp_left_indexes.size() == 0) // ������ӽڵ�Ϊ��
			{
				var_lc = 0.0;
			}
			else
			{
				var_lc = Ex_2_lc / tmp_left_indexes.size() - pow(Ex_lc / tmp_left_indexes.size(), 2)
					+ Ey_2_lc / tmp_left_indexes.size() - pow(Ey_lc / tmp_left_indexes.size(), 2);
			}

			if (tmp_right_indexes.size() == 0) // ������ӽڵ�Ϊ��
			{
				var_rc = 0.0;
			}
			else
			{
				var_rc = Ex_2_rc / tmp_right_indexes.size() - pow(Ex_rc / tmp_right_indexes.size(), 2)
					+ Ey_2_rc / tmp_right_indexes.size() - pow(Ey_rc / tmp_right_indexes.size(), 2);
			}
			// ���㷽������
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
			node->_is_leaf = true; // the node can contain all the samples
			return 1; // ����Ҷ�ӽڵ�
		}
		node->_threshold = threshold; // f������ֵ
		//node->thre_changed_ = true;
		node->_feature_locations = _local_position[feature_index]; // �������ֵ
		selected_indexes.insert(feature_index);
		return 0;
	}
	return -1;
}


void RandomForest::SaveRandomForest(std::ofstream& fout){
	fout << _stage << " "
		<< _local_features_num << " "
		<< _landmark_index << " "
		<< _tree_depth << " "
		<< _trees_num_per_forest << " "
		<< _local_radius << " "
		<< _all_leaf_nodes << " "
		<< _trees.size() << std::endl;
	for (int i = 0; i < _trees.size(); i++){
		Node* root = _trees[i];
		WriteTree(root, fout);
	}
}


void RandomForest::WriteTree(Node* p, std::ofstream& fout){
	if (!p){
		fout << "#" << std::endl;
	}
	else{
		fout << "Y" << " "
			<< p->_threshold << " "
			<< p->_is_leaf << " "
			<< p->_leaf_identity << " "
			<< p->_depth << " "
			<< p->_feature_locations.start.x << " "
			<< p->_feature_locations.start.y << " "
			<< p->_feature_locations.end.x << " "
			<< p->_feature_locations.end.y << std::endl;
		WriteTree(p->_left_child, fout);
		WriteTree(p->_right_child, fout);
	}
}

void RandomForest::LoadRandomForest(std::ifstream& fin){

	int tree_size;
	fin >> _stage
		>> _local_features_num
		>> _landmark_index
		>> _tree_depth
		>> _trees_num_per_forest
		>> _local_radius
		>> _all_leaf_nodes
		>> tree_size;
	std::string start_flag;
	_trees.clear();
	for (int i = 0; i < tree_size; i++){
		Node* root = ReadTree(fin);
		_trees.push_back(root);
	}
}

Node* RandomForest::ReadTree(std::ifstream& fin){
	std::string flag;
	fin >> flag;
	if (flag == "Y"){
		Node* p = new Node();
		fin >> p->_threshold
			>> p->_is_leaf
			>> p->_leaf_identity
			>> p->_depth
			>> p->_feature_locations.start.x
			>> p->_feature_locations.start.y
			>> p->_feature_locations.end.x
			>> p->_feature_locations.end.y;
		p->_left_child = ReadTree(fin);
		p->_right_child = ReadTree(fin);
		return p;
	}
	else{
		return NULL;
	}
}