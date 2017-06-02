#define _CRT_SECURE_NO_WARNINGS
#include "LBF.h"
#include "regressor.h"
using namespace cv;
using namespace std;

void TestImage(Mat& img, CascadeRegressor& rg){
	extern string cascadeName;
	cv::CascadeClassifier haar_cascade;
	bool yes = haar_cascade.load(cascadeName); // 加载人脸检测器
	double scale = 1.3;
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);//灰度和灰度直方图均衡化的图片

	cvtColor(img, gray, CV_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);
	std::vector<cv::Rect> faces;

	haar_cascade.detectMultiScale(smallImg, faces, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(50, 50));

	for (int i = 0; i < faces.size(); i++){
		cv::Rect faceRec = faces[i];
		BoundingBox bbox;
		bbox.start_x = faceRec.x*scale + 0.0578 * faceRec.width*scale;
		bbox.start_y = faceRec.y*scale + 0.2166 * faceRec.height*scale;
		bbox.width = faceRec.width*scale  * 0.8924;
		bbox.height = faceRec.height*scale * 0.8676;
		bbox.centroid_x = bbox.start_x + bbox.width / 2.0;
		bbox.centroid_y = bbox.start_y + bbox.height / 2.0;
		cv::Mat_<double> current_shape = ReProjectShape(global_params._mean_shape, bbox);
		cv::Mat_<double> res = rg.Predict(gray, current_shape, bbox);//, ground_truth_shapes[i]);

		// draw bounding box
		rectangle(img, Point(bbox.start_x, bbox.start_y),
			Point(bbox.start_x + bbox.width, bbox.start_y + bbox.height), Scalar(0, 255, 0), 1, 8, 0);
		// draw result :: red
		for (int i = 0; i < res.rows; i++){
			circle(img, Point2d(res(i, 0), res(i, 1)), 2, Scalar(0, 0, 0), 3, 8, 0);
			circle(img, cv::Point2f(res(i, 0), res(i, 1)), 1, Scalar(255, 255, 255), 2, 8, 0);
			//cout << shape(i, 0)<<" " << shape(i, 1) << endl;
		}
	}
	cv::imshow("result", img);
}


int FaceDetectionAndAlignment(const char* name)
{
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(modelPath + "LBF.model");

	string inputName;//图片
	cv::VideoCapture capture = 0;
	Mat frame, frameCopy, image;
	if (name != NULL){
		inputName.assign(name);
	}
	// name is empty or a number摄像头模式
	if (inputName.empty())
	{
		capture.open(0);
		if (!capture.isOpened()){
			int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0';
			cout << "Capture from CAM " << c << " didn't work" << endl;
			return -1;
		}
	}
	// name is not empty数据模式
	else if (inputName.size())
	{
		if (inputName.find(".jpg") != string::npos || inputName.find(".png") != string::npos
			|| inputName.find(".bmp") != string::npos){//图片
			image = imread(inputName, 1);
			if (image.empty()){
				cout << "Read Image fail" << endl;

			}
		}
		else if (inputName.find(".mp4") != string::npos || inputName.find(".avi") != string::npos
			|| inputName.find(".wmv") != string::npos){//视频
			capture.open(inputName.c_str());
			if (!capture.isOpened())
			{
				cout << "Capture from AVI didn't work" << endl;
			}
		}
	}

	cv::namedWindow("result", 1);
	if (capture.isOpened())//视频模式
	{
		double rate = capture.get(CV_CAP_PROP_FPS);
		bool stop(false);
		int delay = 1000 / rate;
		cout << "In capture ..." << endl;
		while (!stop)
		{
			capture >> frame;
			if (frame.empty())
				break;
			else
				TestImage(frameCopy, cas_load);
			if (waitKey(delay) >= 0)
				stop = true;
		}
		waitKey(0);
		capture.release();
	}
	else
	{
		if (!image.empty())
		{//单个图片模式
			cout << "In image read" << endl;
			TestImage(image, cas_load);
			waitKey(0);
		}
		else if (!inputName.empty()) // 数据集模式
		{
			/* assume it is a text file containing the
			list of the image filenames to be processed - one per line */
			cout << "In image set model" << endl;
			FILE* f = fopen(inputName.c_str(), "rt");
			if (f){
				char buf[1000 + 1];
				while (fgets(buf, 1000, f)){
					int len = (int)strlen(buf), c;
					while (len > 0 && isspace(buf[len - 1]))
						len--;
					buf[len] = '\0';
					cout << "file " << buf << endl;
					image = imread(buf, 1);
					if (!image.empty()){
						TestImage(image, cas_load);
						c = waitKey(0);
						if (c == 27 || c == 'q' || c == 'Q')
							break;
					}
					else{
						cerr << "Aw snap, couldn't read image " << buf << endl;
					}
				}
				fclose(f);
			}
		}
	}
	return 0;
}