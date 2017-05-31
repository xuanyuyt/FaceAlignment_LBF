#ifndef LBF_H
#define LBF_H

# include "opencv2/highgui.hpp"
# include "opencv2/imgproc.hpp"
# include "opencv2/core.hpp"

class Parameters{
public:
	double _bagging_overlap; // 随机森林Bagging参数
	int _trees_num; // 随机森林树的数目
	int _tree_depth; // 每颗树的最大深度
	int _landmarks_num; // landmarks 点数
	int _initial_num; // 多初始形状

	int _regressor_stages; // 回归级数
	std::vector<double> _local_radius; // 局部采样半径
	//std::vector<int> _local_features_num; // 局部采样点数
	int _local_features_num; // 局部采样点数
	cv::Mat_<double> _mean_shape; // 平均形状
};


#endif // !LBF_H
