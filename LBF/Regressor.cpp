#include "Regressor.h"
using namespace std;


// 级联训练开始
void CascadeRegressor::Train(const std::vector<cv::Mat_<uchar> >& images,
	const std::vector<cv::Mat_<double> >& ground_truth_shapes,
	const std::vector<BoundingBox>& bboxes,
	const Parameters& params)
{
	std::cout << "Start training..." << std::endl;

	// data augmentation and multiple initialization
	vector<int> augmented_images_index;	// 图片索引	// 图片索引
	vector<BoundingBox> augmented_bounding_boxs; //扩充后图片人脸框
	vector<cv::Mat_<double> > augmented_ground_truth_shapes;	//扩充后真值
	vector<cv::Mat_<double> > augmented_current_shapes;	//扩充后当前形状

	cv::RNG random_generator(cv::getTickCount());
	for (int i = 0; i < images.size(); i++){
		for (int j = 0; j < global_params._initial_num; j++){
			int index = 0;
			do{
				index = random_generator.uniform(0, (int)images.size());
			} while (index == i);

			// 1. Select ground truth shapes of other images as initial shapes
			augmented_images_index.push_back(i);
			augmented_ground_truth_shapes.push_back(ground_truth_shapes[i]);
			augmented_bounding_boxs.push_back(bboxes[i]);

			// 2. Project current shape to bounding box of ground truth shapes
			cv::Mat_<double> temp = ProjectShape(ground_truth_shapes[index], bboxes[index]);
			temp = ReProjectShape(temp, bboxes[i]);
			augmented_current_shapes.push_back(temp);
		}
	}
	std::cout << "augmented size: " << augmented_current_shapes.size() << std::endl;

	std::vector<cv::Mat_<double> > shape_increaments; //预测 delta_S
	_regressors.resize(params._regressor_stages);
	for (int i = 0; i < params._regressor_stages; i++) //回归级数
	{
		cout << "training stage: " << i + 1 << " of " << params._regressor_stages << endl;
		shape_increaments = _regressors[i].Train(images,
			augmented_images_index,
			augmented_ground_truth_shapes,
			augmented_bounding_boxs,
			augmented_current_shapes,
			params,
			i);//训练形状增量

		std::cout << "update current shapes" << std::endl;
		for (int j = 0; j < shape_increaments.size(); j++)//更新形状
		{
			augmented_current_shapes[j] = shape_increaments[j] + ProjectShape(augmented_current_shapes[j], augmented_bounding_boxs[j]);
			augmented_current_shapes[j] = ReProjectShape(augmented_current_shapes[j], augmented_bounding_boxs[j]);
		}
	}
}


vector<cv::Mat_<double> > Regressor::Train(const vector<cv::Mat_<uchar> >& images,
	const vector<int>& augmented_images_index,
	const vector<cv::Mat_<double> >& augmented_ground_truth_shapes,
	const vector<BoundingBox>& augmented_bboxes,
	const vector<cv::Mat_<double> >& augmented_current_shapes,
	const Parameters& params,
	const int stage)
{
	_stage = stage;
	vector<cv::Mat_<double> > regression_targets; // 回归目标
	vector<cv::Mat_<double> > rotations; // 相似变换旋转矩阵
	vector<double> scales; // 相似变换缩放因子
	regression_targets.resize(augmented_current_shapes.size());
	rotations.resize(augmented_current_shapes.size());
	scales.resize(augmented_current_shapes.size());

	// calculate the regression targets
	std::cout << "calculate regression targets" << std::endl;
	for (int i = 0; i < augmented_current_shapes.size(); i++){//计算回归目标（形状增量）
		regression_targets[i] = ProjectShape(augmented_ground_truth_shapes[i], augmented_bboxes[i])
			- ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]);//回归目标归一化
		cv::Mat_<double> rotation;
		double scale;
		SimilarityTransform(params._mean_shape, ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]), rotation, scale);//初始形状向平均形状对齐
		cv::transpose(rotation, rotation);//转置矩阵？？
		regression_targets[i] = scale * regression_targets[i] * rotation;
		SimilarityTransform(ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]), params._mean_shape, rotation, scale);//平均形状向初始形状对齐
		rotations[i] = rotation;
		scales[i] = scale;
	}

	/* 训练随机森林 */
	std::cout << "train forest of stage:" << _stage + 1 << std::endl;
	_rd_forests.resize(params._landmarks_num_per_face);
	for (int i = 0; i < params._landmarks_num_per_face; ++i) //对每个特征点
	{
		std::cout << "landmark: " << i << std::endl;
		_rd_forests[i] = RandomForest(params, i, _stage, regression_targets);//初始化参数
		_rd_forests[i].TrainForest(//训练随机森林
			images, augmented_images_index, augmented_bboxes, augmented_current_shapes,
			rotations, scales);
	}
	/* 特征编码 */
	std::cout << "Get Global Binary Features" << std::endl;
	struct feature_node **global_binary_features;
	global_binary_features = new struct feature_node*[augmented_current_shapes.size()];

	for (int i = 0; i < augmented_current_shapes.size(); ++i)
	{
		global_binary_features[i] = new feature_node[params._trees_num_per_forest * params._landmarks_num_per_face + 1];//+1 是标识符
	}
	int num_feature = 0;//随机森林所有叶子节点数
	for (int i = 0; i < params._landmarks_num_per_face; ++i){
		num_feature += _rd_forests[i]._all_leaf_nodes;// landmarks×2^depth
	}

	for (int i = 0; i < augmented_current_shapes.size(); ++i)
	{
		int index = 1; // 叶子节点索引，从1开始
		int ind = 0; // 序号 0 ~ landmarks×trees-1
		const cv::Mat_<double>& rotation = rotations[i];
		const double scale = scales[i];
		const cv::Mat_<uchar>& image = images[augmented_images_index[i]];
		const BoundingBox& bbox = augmented_bboxes[i];
		const cv::Mat_<double>& current_shape = augmented_current_shapes[i];

		for (int j = 0; j < params._landmarks_num_per_face; ++j) // 每个特征点
		{
			for (int k = 0; k < params._trees_num_per_forest; ++k) // 每棵树
			{
				Node* node = _rd_forests[j]._trees[k];
				while (!node->_is_leaf) // 非叶节点
				{
					FeatureLocations& pos = node->_feature_locations; // 最优分裂特征索引
					double delta_x = rotation(0, 0)*pos.start.x + rotation(0, 1)*pos.start.y;
					double delta_y = rotation(1, 0)*pos.start.x + rotation(1, 1)*pos.start.y;
					delta_x = scale*delta_x*bbox.width / 2.0;
					delta_y = scale*delta_y*bbox.height / 2.0;
					int real_x = delta_x + current_shape(j, 0);
					int real_y = delta_y + current_shape(j, 1);
					real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
					real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
					int tmp = (int)image(real_y, real_x); //real_y at first

					delta_x = rotation(0, 0)*pos.end.x + rotation(0, 1)*pos.end.y;
					delta_y = rotation(1, 0)*pos.end.x + rotation(1, 1)*pos.end.y;
					delta_x = scale*delta_x*bbox.width / 2.0;
					delta_y = scale*delta_y*bbox.height / 2.0;
					real_x = delta_x + current_shape(j, 0);
					real_y = delta_y + current_shape(j, 1);
					real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
					real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
					// 节点分裂
					if ((tmp - (int)image(real_y, real_x)) < node->_threshold)
					{
						node = node->_left_child;// go left
					}
					else
					{
						node = node->_right_child;// go right
					}
				}
				//样本i在第ind颗树上落到的叶子节点序号，从1开始(1-25; 26-50; ...)
				global_binary_features[i][ind].index = index + node->_leaf_identity;// 
				global_binary_features[i][ind].value = 1.0;
				ind++;
			}
			index += _rd_forests[j]._all_leaf_nodes;
		}
		if (i % 500 == 0 && i > 0){
			cout << "extracted " << i << " images" << endl;
		}
		global_binary_features[i][params._trees_num_per_forest*params._landmarks_num_per_face].index = -1; //标记编码结束
		global_binary_features[i][params._trees_num_per_forest*params._landmarks_num_per_face].value = -1.0;//标记编码结束
	}

	/* 训练回归器 */
	struct problem* prob = new struct problem;
	prob->l = augmented_current_shapes.size();
	prob->n = num_feature;
	prob->x = global_binary_features;
	prob->bias = -1;

	struct parameter* regression_params = new struct parameter;
	regression_params->solver_type = L2R_L2LOSS_SVR_DUAL;
	regression_params->C = 1.0 / augmented_current_shapes.size();
	regression_params->p = 0;
	//regression_params->eps = 0.0001;

	std::cout << "Global Regression of stage " << _stage << std::endl;
	_linear_model_x.resize(params._landmarks_num_per_face);
	_linear_model_y.resize(params._landmarks_num_per_face);
	double* targets = new double[augmented_current_shapes.size()];
	for (int i = 0; i < params._landmarks_num_per_face; ++i) //回归每个landmark
	{
		std::cout << "regress landmark " << i << std::endl;
		for (int j = 0; j< augmented_current_shapes.size(); j++){
			targets[j] = regression_targets[j](i, 0);
		}
		prob->y = targets;
		check_parameter(prob, regression_params);
		struct model* regression_model = train(prob, regression_params);
		_linear_model_x[i] = regression_model;
		for (int j = 0; j < augmented_current_shapes.size(); j++){
			targets[j] = regression_targets[j](i, 1);
		}
		prob->y = targets;
		check_parameter(prob, regression_params);
		regression_model = train(prob, regression_params);
		_linear_model_y[i] = regression_model;
	}

	std::cout << "predict regression targets" << std::endl;
	std::vector<cv::Mat_<double> > predict_regression_targets;
	predict_regression_targets.resize(augmented_current_shapes.size());
	for (int i = 0; i < augmented_current_shapes.size(); i++){
		cv::Mat_<double> atargets_predict(params._landmarks_num_per_face, 2, 0.0);
		for (int j = 0; j < params._landmarks_num_per_face; j++){
			atargets_predict(j, 0) = predict(_linear_model_x[j], global_binary_features[i]);
			atargets_predict(j, 1) = predict(_linear_model_y[j], global_binary_features[i]);
		}
		cv::Mat_<double> rot;
		cv::transpose(rotations[i], rot);//转置矩阵？？
		predict_regression_targets[i] = scales[i] * atargets_predict * rot;
		if (i % 500 == 0 && i > 0){
			std::cout << "predict " << i << " images" << std::endl;
		}
	}
	delete[] targets;
	for (int i = 0; i< augmented_current_shapes.size(); i++)
	{
		delete[] global_binary_features[i];
	}
	delete[] global_binary_features;

	return predict_regression_targets;
}

void CascadeRegressor::SaveCascadeRegressor(std::string ModelName){
	std::ofstream fout;

	fout.open((ModelName + "_params").c_str(), std::fstream::out);
	fout << _params._local_features_num << " "
		<< _params._landmarks_num_per_face << " "
		<< _params._regressor_stages << " "
		<< _params._tree_depth << " "
		<< _params._trees_num_per_forest << " "
		<< _params._initial_num << std::endl;
	for (int i = 0; i < _params._regressor_stages; i++){
		fout << _params._local_radius[i] << std::endl;
	}
	for (int i = 0; i < _params._landmarks_num_per_face; i++){
		fout << _params._mean_shape(i, 0) << " " << _params._mean_shape(i, 1) << std::endl;
	}

	fout.close();

	fout.open(ModelName + "_regressors", ios::binary);

	for (int i = 0; i < _params._regressor_stages; i++){
		_regressors[i].SaveRegressor(fout);
		fout << _Models[i].size() << endl;
		for (int j = 0; j<_Models[i].size(); j++){
			save_model_bin(fout, _Models[i][j]);
		}
		
	}
	fout.close();
}

void CascadeRegressor::LoadCascadeRegressor(std::string ModelName){
	std::ifstream fin;
	fin.open((ModelName + "_params").c_str(), std::fstream::in);
	_params = Parameters();
	fin >> _params._local_features_num
		>> _params._landmarks_num_per_face
		>> _params._regressor_stages
		>> _params._tree_depth
		>> _params._trees_num_per_forest
		>> _params._initial_num;

	std::vector<double> local_radius_by_stage;
	local_radius_by_stage.resize(_params._regressor_stages);
	for (int i = 0; i < _params._regressor_stages; i++){
		fin >> local_radius_by_stage[i];
	}
	_params._local_radius = local_radius_by_stage;

	cv::Mat_<double> mean_shape(_params._landmarks_num_per_face, 2, 0.0);
	for (int i = 0; i < _params._landmarks_num_per_face; i++){
		fin >> mean_shape(i, 0) >> mean_shape(i, 1);
	}
	_params._mean_shape = mean_shape;

	fin.close();
	fin.open((ModelName + "_regressors").c_str(), std::fstream::in);
	_regressors.resize(_params._regressor_stages);
	for (int i = 0; i < _params._regressor_stages; i++){
		_regressors[i]._params = _params;
		_regressors[i].LoadRegressor(fin);
		_regressors[i].ConstructLeafCount();
		int num = 0;
		fin >> num;
		_Models[i].resize(num);
		for (int j = 0; j<num; j++){
			_Models[i][j] = load_model_bin(fin);
		}

	}
}

void Regressor::SaveRegressor(std::ofstream& fout)
{
	fout << _stage << " "
		<< _rd_forests.size() << " "
		<< _linear_model_x.size() << std::endl;

	for (int i = 0; i < _rd_forests.size(); i++){
		_rd_forests[i].SaveRandomForest(fout);
	}

}

void Regressor::LoadRegressor(std::ifstream& fin)
{
	int rd_size, linear_size;
	fin >> _stage >> rd_size >> linear_size;
	_rd_forests.resize(rd_size);
	for (int i = 0; i < rd_size; i++){
		_rd_forests[i].LoadRandomForest(fin);
	}

}


void Regressor::ConstructLeafCount(){
	int index = 1;
	for (int i = 0; i < _params._landmarks_num_per_face; ++i){
		leaf_index_count[i] = index;
		index += _rd_forests[i]._all_leaf_nodes;
	}
}