#ifndef LBF_H
#define LBF_H

# include "opencv2/highgui.hpp"
# include "opencv2/imgproc.hpp"
# include "opencv2/core.hpp"

class Parameters{
public:
	double _bagging_overlap; // ���ɭ��Bagging����
	int _trees_num; // ���ɭ��������Ŀ
	int _tree_depth; // ÿ������������
	int _landmarks_num; // landmarks ����
	int _initial_num; // ���ʼ��״

	int _regressor_stages; // �ع鼶��
	std::vector<double> _local_radius; // �ֲ������뾶
	//std::vector<int> _local_features_num; // �ֲ���������
	int _local_features_num; // �ֲ���������
	cv::Mat_<double> _mean_shape; // ƽ����״
};


#endif // !LBF_H
