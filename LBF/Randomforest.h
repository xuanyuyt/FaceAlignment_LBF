#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "LBF.h"
#include <set>

class Node {
public:
	int _leaf_identity; // 叶子节点处序号 0 ~ 2^depth-1
	Node* _left_child; // 左子节点
	Node* _right_child; // 右子节点
	int _samples; // 样本容量
	bool _is_leaf; // 是否为叶子节点
	int _depth; // 当前节点深度
	double _threshold; // 分裂阈值 
	FeatureLocations _feature_locations; // 最优分裂像素差特征的相对索引
	Node(Node* left, Node* right, double thres, bool leaf);
	Node(Node* left, Node* right, double thres);
	Node();
};

class RandomForest {
public:
	int _stage;
	int _local_features_num;
	int _landmark_index;
	int _tree_depth; // 最大深度
	int _trees_num_per_forest;
	double _local_radius;
	int _all_leaf_nodes;
	std::vector<cv::Mat_<double> >* _regression_targets; // 回归目标
	std::vector<FeatureLocations> _local_position; // size = param_.local_features_num
	std::vector<Node*> _trees;


	RandomForest(){};
	RandomForest(const Parameters& param, int landmark_index, int stage, std::vector<cv::Mat_<double> >& regression_targets);
	bool TrainForest(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<int>& augmented_images_index,
		const std::vector<BoundingBox>& augmented_bboxes,
		const std::vector<cv::Mat_<double> >& augmented_current_shapes,
		const std::vector<cv::Mat_<double> >& rotations,
		const std::vector<double>& scales);
	// 训练一颗决策树
	Node* BuildTree(std::set<int>& selected_indexes, cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, int current_depth);
	// 节点分裂
	int FindSplitFeature(Node* node, std::set<int>& selected_indexes,
		cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int>& left_indexes, std::vector<int>& right_indexes);
};


#endif