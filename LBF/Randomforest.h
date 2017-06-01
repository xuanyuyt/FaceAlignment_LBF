#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "LBF.h"
#include <set>

class Node {
public:
	int _leaf_identity; // Ҷ�ӽڵ㴦��� 0 ~ 2^depth-1
	Node* _left_child; // ���ӽڵ�
	Node* _right_child; // ���ӽڵ�
	int _samples; // ��������
	bool _is_leaf; // �Ƿ�ΪҶ�ӽڵ�
	int _depth; // ��ǰ�ڵ����
	double _threshold; // ������ֵ 
	FeatureLocations _feature_locations; // ���ŷ������ز��������������
	Node(Node* left, Node* right, double thres, bool leaf);
	Node(Node* left, Node* right, double thres);
	Node();
};

class RandomForest {
public:
	int _stage;
	int _local_features_num;
	int _landmark_index;
	int _tree_depth; // ������
	int _trees_num_per_forest;
	double _local_radius;
	int _all_leaf_nodes;
	std::vector<cv::Mat_<double> >* _regression_targets; // �ع�Ŀ��
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
	// ѵ��һ�ž�����
	Node* BuildTree(std::set<int>& selected_indexes, cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, int current_depth);
	// �ڵ����
	int FindSplitFeature(Node* node, std::set<int>& selected_indexes,
		cv::Mat_<int>& pixel_differences, std::vector<int>& images_indexes, std::vector<int>& left_indexes, std::vector<int>& right_indexes);
};


#endif