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
	vector<Mat_<uchar> > augmented_images;	// ͼƬ����
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
			augmented_images.push_back(images[i]);
			augmented_ground_truth_shapes.push_back(ground_truth_shapes[i]);
			augmented_bounding_boxs.push_back(bboxes[i]);

			// 2. Project current shape to bounding box of ground truth shapes
			Mat_<double> temp = ProjectShape(ground_truth_shapes[index], bboxes[index]);
			temp = ReProjectShape(temp, bboxes[i]);
			augmented_current_shapes.push_back(temp);
		}
	}

}