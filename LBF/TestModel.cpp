#include "LBF.h"
#include "regressor.h"
using namespace cv;
using namespace std;


void TestModel(vector<string> testDataName)
{
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(modelPath + "LBF.model");
	std::vector<cv::Mat > test_images;
	std::vector<cv::Mat_<uchar> > images_gray;
	std::vector<cv::Mat_<double> > ground_truth_shapes;
	std::vector<BoundingBox> bounding_boxs;
	for (int i = 0; i<testDataName.size(); i++)
	{
		string path;
		if (testDataName[i] == "COFW")
		{
			//LoadCofwTestData(test_images, images_gray, ground_truth_shapes, bounding_boxs);
			break;
		}
		else if (testDataName[i] == "HELEN" || testDataName[i] == "LFPW")
		{
			path = dataPath + testDataName[i] + "/testset/Path_Images.txt";
		}
		else
			path = dataPath + testDataName[i] + "/Path_Images.txt";

		LoadData(path, test_images, images_gray, ground_truth_shapes, bounding_boxs);
		//LoadOpencvBbxData(path, test_images, ground_truth_shapes, bounding_boxs);
	}

	double MRSE_sum = 0;
	for (int i = 0; i < images_gray.size(); i++){
		cv::Mat_<double> current_shape = ReProjectShape(global_params._mean_shape, bounding_boxs[i]);
		cv::Mat_<double> res = cas_load.Predict(images_gray[i], current_shape, bounding_boxs[i]);//, ground_truth_shapes[i]);
		double temp = CalculateError68(ground_truth_shapes[i], res);//计算每个图片对其误差，并求和. 68blandmarks
		
		cout << "test " << i << " image" << "  " << temp * 100 << "%" << endl;
		MRSE_sum += temp;

		DrawPredictedImage(test_images[i], res);
		waitKey(0);
	}

	double MRSE = MRSE_sum / ground_truth_shapes.size();//平均对其误差
	cout << "test erro :" << MRSE * 100 << "%" << endl;
	return;
}