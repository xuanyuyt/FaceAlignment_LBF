#include "Regressor.h"
using namespace std;
using namespace cv;


// ����ѵ����ʼ
void CascadeRegressor::Train(const std::vector<cv::Mat_<uchar> >& images,
	const std::vector<cv::Mat_<double> >& ground_truth_shapes,
	const std::vector<BoundingBox>& bboxes,
	const Parameters& params)
{
	std::cout << "Start training..." << std::endl;

	// data augmentation and multiple initialization
	vector<int> augmented_images_index;	// ͼƬ����	// ͼƬ����
	vector<BoundingBox> augmented_bounding_boxs; //�����ͼƬ������
	vector<Mat_<double> > augmented_ground_truth_shapes;	//�������ֵ
	vector<Mat_<double> > augmented_current_shapes;	//�����ǰ��״

	RNG random_generator(getTickCount());
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
			Mat_<double> temp = ProjectShape(ground_truth_shapes[index], bboxes[index]);
			temp = ReProjectShape(temp, bboxes[i]);
			augmented_current_shapes.push_back(temp);
		}
	}
	std::cout << "augmented size: " << augmented_current_shapes.size() << std::endl;

	std::vector<cv::Mat_<double> > shape_increaments; //Ԥ�� delta_S
	_regressors.resize(params._regressor_stages);
	for (int i = 0; i < params._regressor_stages; i++) //�ع鼶��
	{
		cout << "training stage: " << i + 1 << " of " << params._regressor_stages << endl;
		shape_increaments = _regressors[i].Train(images,
			augmented_images_index,
			augmented_ground_truth_shapes,
			augmented_bounding_boxs,
			augmented_current_shapes,
			params,
			i);//ѵ����״����

		std::cout << "update current shapes" << std::endl;
		for (int j = 0; j < shape_increaments.size(); j++)//������״
		{
			augmented_current_shapes[j] = shape_increaments[j] + ProjectShape(augmented_current_shapes[j], augmented_bounding_boxs[j]);
			augmented_current_shapes[j] = ReProjectShape(augmented_current_shapes[j], augmented_bounding_boxs[j]);
		}
	}
}


vector<Mat_<double> > Regressor::Train(const vector<Mat_<uchar> >& images,
	const vector<int>& augmented_images_index,
	const vector<Mat_<double> >& augmented_ground_truth_shapes,
	const vector<BoundingBox>& augmented_bboxes,
	const vector<Mat_<double> >& augmented_current_shapes,
	const Parameters& params,
	const int stage)
{
	_stage = stage;
	vector<Mat_<double> > regression_targets; // �ع�Ŀ��
	vector<Mat_<double> > rotations; // ���Ʊ任��ת����
	vector<double> scales; // ���Ʊ任��������
	regression_targets.resize(augmented_current_shapes.size());
	rotations.resize(augmented_current_shapes.size());
	scales.resize(augmented_current_shapes.size());

	// calculate the regression targets
	std::cout << "calculate regression targets" << std::endl;
	for (int i = 0; i < augmented_current_shapes.size(); i++){//����ع�Ŀ�꣨��״������
		regression_targets[i] = ProjectShape(augmented_ground_truth_shapes[i], augmented_bboxes[i])
			- ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]);//�ع�Ŀ���һ��
		cv::Mat_<double> rotation;
		double scale;
		SimilarityTransform(params._mean_shape, ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]), rotation, scale);//��ʼ��״��ƽ����״����
		cv::transpose(rotation, rotation);//ת�þ��󣿣�
		regression_targets[i] = scale * regression_targets[i] * rotation;
		SimilarityTransform(ProjectShape(augmented_current_shapes[i], augmented_bboxes[i]), params._mean_shape, rotation, scale);//ƽ����״���ʼ��״����
		rotations[i] = rotation;
		scales[i] = scale;
	}

	std::cout << "train forest of stage:" << _stage + 1 << std::endl;
	_rd_forests.resize(params._landmarks_num);
	for (int i = 0; i < params._landmarks_num; ++i) //��ÿ��������
	{
		std::cout << "landmark: " << i << std::endl;
		_rd_forests[i] = RandomForest(params, i, _stage, regression_targets);//��ʼ������
		_rd_forests[i].TrainForest(//ѵ�����ɭ��
			images, augmented_images_index, augmented_bboxes, augmented_current_shapes,
			rotations, scales);
	}

	std::cout << "Get Global Binary Features" << std::endl;


	std::cout << "Global Regression of stage " << _stage << std::endl;


	std::cout << "predict regression targets" << std::endl;
	std::vector<cv::Mat_<double> > predict_regression_targets;

	return predict_regression_targets;
}