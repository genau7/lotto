#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <math.h>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define R 2
#define G 1
#define B 0
#define H 0
#define S 1
#define V 2
#define ABS(x) ((x>0)? x : -x)

typedef std::vector<std::vector<uchar>> BinaryArray;
const int n = 4;
const int dilateSEsize = 7;
const int erodeSEsize = 5;
const int SE7[7][7] =
{
	{ 0, 0, 1, 1, 1, 0, 0 },
	{ 0, 1, 1, 1, 1, 1, 0 },
	{ 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 1, 1, 1, 1, 1, 1 },
	{ 0, 1, 1, 1, 1, 1, 0 },
	{ 0, 0, 1, 1, 1, 0, 0 }
};
/*/
const int dilateSE[dilateSEsize][dilateSEsize] =
{
{ 0, 1, 1, 1, 0 },
{ 1, 1, 1, 1, 1 },
{ 1, 1, 1, 1, 1 },
{ 1, 1, 1, 1, 1,},
{ 0, 1, 1, 1, 0 }
};

/*
const int SE3[3][3] =
{
	{ 0, 1, 0 },
	{ 1, 1, 1 },
	{ 0, 1, 0 }
};
/*const int SE3[3][3] =
{
	{ 1, 1, 1 },
	{ 1, 1, 1 },
	{ 1, 1, 1 }
};*/
const int SE3[3][3] =
{
	{ 0, 1, 0 },
	{ 1, 1, 1 },
	{ 0, 1, 0 }
};
const int SE5[5][5] =
{
	{ 0, 1, 1, 1, 0 },
	{ 1, 1, 1, 1, 1 },
	{ 1, 1, 1, 1, 1 },
	{ 1, 1, 1, 1, 1 },
	{ 0, 1, 1, 1, 0 }
};

const int kernelSize = 3;
const int maskH[kernelSize][kernelSize] = { { -1, -1, -1 },
											 { -1,  9, -1 },
											 { -1, -1, -1 } }; 

const int maskL[kernelSize][kernelSize] = { { 1, 1, 1 },
											{ 1, 1, 1 },
											{ 1, 1, 1 } }; 

int window[7][7] = { 0 };
cv::Mat originalImgs[n];
cv::Mat source;
char windowName[30] = "HSV vals";

struct Pixel {
	int x;
	int y;
};

class MomentFinder{
public:
	BinaryArray* mask;
	int rows, cols;
	Pixel massCenter;
	int area;
	MomentFinder(BinaryArray* mask, int rows, int cols){
		this->mask = mask;
		this->rows = rows;
		this->cols = cols;
	}
	void calcArea(int* area){
		*area = m(0, 0);
		this->area = *area;
	}
	void calcMassCenter(Pixel* massCenter){
		this->massCenter.y = m(1, 0) / area; //row
		this->massCenter.x = m(0, 1) / area; //col
		massCenter = &(this->massCenter);
	}

	float calcM7(){
		return (M(2, 0)*M(0, 2) - pow(M(1, 1), 2)) / (pow(area, 4));
	}

	float m(int p, int q){
		float m = 0;
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				if (mask->at(i).at(j) != 0)
					m += pow(i, p)*pow(j, q);
		return m;
	}

	float M(int p, int q){
		float M = 0;
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				if (mask->at(i).at(j) != 0)
					M += pow((i - massCenter.y), p)*pow((j - massCenter.x), q);
		return M;
	}

};

class Segment{
public:
	Segment(int index, int minRow, int minCol, int maxRow, int maxCol, int** labels){
		this->index = index;
		this->minRow = minRow;
		this->minCol = minCol;
		this->maxRow = maxRow;
		this->maxCol = maxCol;
		this->labels = labels;
		rows = maxRow - minRow + 1;
		cols = maxCol - minCol + 1;
		for (int i = 0; i < rows; ++i){
			std::vector<uchar> rowVector;
			mask.push_back(rowVector);
			for (int j = 0; j < cols; ++j){
				int label = labels[i+minRow][j+minCol];
				if (label != 0)
					mask[i].push_back(1);
				else
					mask[i].push_back(0);
			}
		}
	}
	void calcParams(){
		MomentFinder momentFinder(&mask, rows, cols);
		momentFinder.calcArea(&area);
		momentFinder.calcMassCenter(&massCenter);
		M7 = momentFinder.calcM7();
	}
	void clear(){
		for (int i = minRow; i <= maxRow; ++i)
			for (int j = minCol; j <= maxCol; ++j){
				labels[i][j] = 0;
			}
	}
	int minRow, minCol;
	int maxRow, maxCol;
	int index;
	int rows, cols;
    Pixel massCenter;
	BinaryArray mask;
	int **labels;
	int area;
	float radius;
	float M7;

};

struct Img {
	cv::Mat originalImg;
	cv::Mat_<cv::Vec3b> binaryFromBlue;
	cv::Mat segmented; 
	int** labels;
	std::string name;
	int rows, cols;
	std::vector<Segment> segments;
	~Img(){
		for (int j = 0; j < cols; ++j)
			delete[] labels[j];
		delete[] labels;
	}
	void init(std::string filename){
		originalImg = cv::imread(filename.c_str());
		name = filename;
		rows = originalImg.rows;
		cols = originalImg.cols;
		cv::Mat temp(originalImg.size(), CV_8U);
		segmented = temp;
		labels = new int*[rows];
		for (int j = 0; j < cols; ++j)
			labels[j] = new int[cols];
	}
	void show(){
		//cv::imshow(name.c_str(), originalImg);
		cv::imshow(name.c_str(), binaryFromBlue);
	}
	void showSegments(bool indexesVisible = false){
		colorSegments();
		if (indexesVisible){
			for (int i = 0; i < segments.size(); ++i){
				char nr[30];
				Segment s = segments.at(i);
				sprintf(nr, "%d", s.index);
				putText(segmented, nr, cv::Point(s.minCol, s.minRow), cv::FONT_HERSHEY_SIMPLEX, .4, cv::Scalar(255, 255, 255), 1, 8, false);
			}
		}
		for (int i = 0; i < segments.size(); ++i){
			char m7[30];
			Segment s = segments.at(i);
			//sprintf(m7, "%.3f", s.M7 * 1000);
			sprintf(m7, "%d", s.area);
			putText(segmented, m7, cv::Point(s.minCol, s.minRow), cv::FONT_HERSHEY_SIMPLEX, .4, cv::Scalar(255, 200, 255), 1, 8, false);
		}
		cv::imshow(name.c_str(), segmented);
	}
	void filterEllipses(){
		for (int k = 0; k < segments.size();){
			Segment s = segments.at(k);
			if (s.M7 > 0.0071 || s.M7 < 0.00638){
				s.clear();
				segments.erase(segments.begin() + k);
			}
			else
				++k;
		}
	}
	void colorSegments(){
		cv::Mat_<uchar> temp = segmented;
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j) {
				temp(i, j) = (labels[i][j] * 10) % 255;
			}
		segmented = temp;
	}
	void labelPixels(){
		int count = 1;
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j) {
				if (binaryFromBlue(i, j)[0] != 0)
					labels[i][j] = count++;
				else
					labels[i][j] = 0;

			}
	}
	bool bottomUpPass(){
		bool change = false;
		int windowSize = 3;
		for (int j = cols - 2; j > 0; --j)
			for (int i = rows - 1; i > 0; --i){
				if (labels[i][j] != 0){
					int min = labels[i][j];
					if (labels[i][j - 1] < min && labels[i][j - 1] > 0)
						min = labels[i][j - 1];
					if (labels[i - 1][j - 1] < min && labels[i - 1][j - 1] > 0)
						min = labels[i - 1][j - 1];
					if (labels[i - 1][j] < min && labels[i - 1][j] > 0)
						min = labels[i - 1][j];
					if (labels[i - 1][j + 1] < min && labels[i - 1][j + 1] > 0)
						min = labels[i - 1][j + 1];
					if (labels[i][j + 1] < min && labels[i][j + 1] > 0)
						min = labels[i][j + 1];

					if (min != labels[i][j]){
						change = true;
						labels[i][j] = min;
					}
				}
			}
		return change;
	}
	bool topDownPass(){
		bool change = false;
		for (int j = 1; j < cols - 1; ++j)
			for (int i = 0; i < rows - 1; ++i){
				if (labels[i][j] != 0){
					int min = labels[i][j];
					if (labels[i][j - 1] < min && labels[i][j - 1] > 0)
						min = labels[i][j - 1];
					if (labels[i + 1][j - 1] < min && labels[i + 1][j - 1] > 0)
						min = labels[i + 1][j - 1];
					if (labels[i + 1][j] < min && labels[i + 1][j] > 0)
						min = labels[i + 1][j];
					if (labels[i + 1][j + 1] < min && labels[i + 1][j + 1] > 0)
						min = labels[i + 1][j + 1];
					if (labels[i][j + 1] < min && labels[i][j + 1] > 0)
						min = labels[i][j + 1];

					if (min != labels[i][j]){
						change = true;
						labels[i][j] = min;
					}
				}
			}
		return change;
	}
	void indexSegments(){
		bool segmentationDone = false;
		while (!segmentationDone){
			bool notFinished1 = topDownPass();
			bool notFinished2 = bottomUpPass();
			segmentationDone = !(notFinished1 || notFinished2);
		}
	}
	void findSegments(){
		labelPixels();
		indexSegments();
		//make a list of all unique segments' indexes
		std::set<int> indexes;
		for (int j = 0; j < cols; ++j)
			for (int i = 0; i < rows ; ++i)
				if (labels[i][j] != 0)
					indexes.insert(labels[i][j]);
		
		std::set<int>::iterator it;
		for (it = indexes.begin(); it != indexes.end(); it++){
			int maxRow = 0, maxCol = 0;
			int minRow = 99999999, minCol = 9999999;
			//find bounding box
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j)
					if (labels[i][j] == *it){
						if (i < minRow)
							minRow = i;
						if (i > maxRow)
							maxRow = i;
						if (j < minCol)
							minCol = j;
						if (j > maxCol)
							maxCol = j;
					}
			Segment segment(*it, minRow, minCol, maxRow, maxCol, labels);
			segments.push_back(segment);
		}
	}
	void calcSegmentsParams(){
		for (int k = 0; k < segments.size(); ++k)
			segments.at(k).calcParams();
	}
};


Img images[n];


struct BoundingBox{
	Pixel upperLeft;
	Pixel lowerRight;
	Pixel center;
	BoundingBox(){
		lowerRight.x = 0;
		lowerRight.y = 0;
		center.x = center.y = 0;
	}

	void calcCenter(){
		center.x = (lowerRight.x - upperLeft.x) / 2;
		center.y = (lowerRight.y - upperLeft.y) / 2;
	}
	cv::Mat_<cv::Vec3b> draw(cv::Mat img){
		cv::Mat_<cv::Vec3b> tempImg = img;
		for (int i = upperLeft.y; i <= lowerRight.y; i++){
			if (i == upperLeft.y || i == lowerRight.y){
				for (int j = upperLeft.x; j < lowerRight.x; j++){
					//if (j == upperLeft.x || j == lowerRight.x){
					if (true){
						tempImg(i, j)[0] = 100;
						tempImg(i, j)[1] = 100;
						tempImg(i, j)[2] = 100;
					}
				}
			}
		}

		for (int i = upperLeft.y; i <= lowerRight.y; i++){
			for (int j = upperLeft.x; j <= lowerRight.x; j++){
				if (j == upperLeft.x || j == lowerRight.x){
						tempImg(i, j)[0] = 100;
						tempImg(i, j)[1] = 100;
						tempImg(i, j)[2] = 100;
				}
			}
		}
		return tempImg;
	}
};


struct Shape {
	BoundingBox box;
	std::string name;
	cv::Mat img;
	cv::Mat boundedImg;
	int area;
	int perim;
	Pixel center;
	float W3, M3, M7, m00;
	float angleR;
	float angle;

	float m(int p, int q){
		float m = 0;
		cv::Mat_<cv::Vec3b> tempImg = boundedImg;
		for (int i = 0; i < boundedImg.rows; i++)
			for (int j = 0; j < boundedImg.cols; j++)
				if (tempImg(i, j)[0] == 0)
					m += pow(i, p)*pow(j, q);
		return m;
	}

	void create(std::string name_in){
		name = name_in;
		img = cv::imread(name.c_str());
		calcBoundingBox();
		area = perim = 0;
		W3 = M3 = M7  = 0.0;
		m00 = m(0, 0);
		center.y = m(1,0) / m00; // i
		center.x = m(0,1) / m00; //j
		//std::cout << "\nm00=" << m00 << ", m01=" << m(0, 1) << std::endl;
	}
	void create(cv::Mat img_in){
		name = "";
		img = img_in;
		calcBoundingBox();
		area = perim = 0;
		W3 = M3 = M7 = 0.0;
		m00 = m(0, 0);
		center.y = m(1, 0) / m00; // i
		center.x = m(0,1) / m00; //j
	}
	void calcAngle(){
		float y = box.center.y - center.y;
		float x = box.center.x - center.x;
		float p = 0;

		//sprawdz cwiartke
		if (x < 0 && y >= 0) //II
			p = 0.5;
		else if (x <= 0 && y <= 0) //III
			p = 1.0;
		else if (x > 0 && y < 0) //IV
			p = 1.5;
		
		if (y == 0 && p == 0.5)
			p = 1.0;

		if (x == 0)
			angleR = 3.14/2;
		else
			angleR = fabs(y) / fabs(x);
		angleR += p * 3.14;
		angle = 180 * angleR / 3.14;
	}
	float N(int p, int q){
		return M(p, q) / pow(m(0, 0), (p + q) / 2 + 1);
	}

	float M(int p, int q){
		float M = 0;
		cv::Mat_<cv::Vec3b> tempImg = boundedImg;
		for (int i = 0; i < boundedImg.rows; i++)
			for (int j = 0; j < boundedImg.cols; j++)
				if (tempImg(i, j)[0] == 0)
					M += pow((i - center.y), p)*pow((j - center.x), q);
		return M;

	}
	void calcM3(){	
		//std::cout << std::endl << N(3, 0) << ", " << N(1, 2) << "," << N(2, 1) << ", " << N(0, 3) << std::endl;
		M3 = pow((N(3,0) - 3 * N(1,2)),2) + pow(3 * N(2,1) - N(0,3),2);
	}

	void calcM7(){
		M7 = N(2, 0)*N(0, 2) - pow(N(1, 1), 2);
	}

	void print(){
		std::cout << name << ":\n" << "\tperimeter=" << perim << "\tarea=" << area <<"\tW3="<<W3<<"\tM3="<<M3<< "\tM7="<<M7<<std::endl;
	}
	void calcBoundingBox(){
		cv::Mat_<cv::Vec3b> tempImg = img;
		box.upperLeft.x = img.cols;
		box.upperLeft.y = img.rows;
		for (int i = 0; i < img.rows; i++){
			for (int j = 0; j < img.cols; j++){
				if (tempImg(i, j)[0] == 0){
					if (j < box.upperLeft.x)
						box.upperLeft.x = j;
					if (i < box.upperLeft.y)
						box.upperLeft.y = i;
					if (j > box.lowerRight.x)
						box.lowerRight.x = j;
					if (i > box.lowerRight.y)
						box.lowerRight.y = i;
				}
			}
		}
		//img = box.draw(img);	
		
		box.calcCenter();
		m00 = m(0, 0);
		center.y = m(0, 1) / m00; // i
		center.x = m(1, 0) / m00; //j

		boundedImg = img(cv::Rect(box.upperLeft.x, box.upperLeft.y, box.lowerRight.x - box.upperLeft.x, box.lowerRight.y - box.upperLeft.y));
	}
	void calcArea(){
		cv::Mat_<cv::Vec3b> tempImg = boundedImg;
		for (int i = 0; i < boundedImg.rows; i++)
			for (int j = 0; j < boundedImg.cols; j++)
				if (tempImg(i, j)[0] == 0)
					area += 1;
	}
	void drawCenters(){
		cv::Mat_<cv::Vec3b> tempImg = boundedImg;
		tempImg(center.y, center.x)[1] = 255;
		tempImg(box.center.y, box.center.x)[2] = 180;
		boundedImg = tempImg;
	}
	void calcPerimeter(){
		cv::Mat_<cv::Vec3b> tempImg = img;
		for (int i = 1; i < img.rows - 1; i++){
			for (int j = 1; j < img.cols - 1; j++){
				if (tempImg(i, j)[0] == 0)
					if (tempImg(i - 1, j - 1)[0] != 0 || tempImg(i - 1, j)[0] != 0 || tempImg(i - 1, j + 1)[0] != 0 ||
						tempImg(i, j - 1)[0] != 0 || tempImg(i, j)[0] != 0 || tempImg(i, j + 1)[0] != 0 ||
						tempImg(i + 1, j - 1)[0] != 0 || tempImg(i + 1, j)[0] != 0 || tempImg(i + 1, j + 1)[0] != 0)
						perim += 1;
			}
		}
	}
	void calcW3(){
		calcArea();
		calcPerimeter();
		W3 = perim / 2 / sqrt(3.14*area) - 1;
	}
};
struct Shape shapes[5];

cv::Mat& perform(cv::Mat& I){
  CV_Assert(I.depth() != sizeof(uchar));
  cv::Mat_<cv::Vec3b> _I = I;
  int half = I.rows / 2;
  for (int i = 0; i < I.rows; ++i){
	  for (int j = 0; j < I.cols; ++j){
		  if (i < half && j < half)
			  ;//NOP
		  if (i >= half && j < half)
			  _I(i, j)[0] = 0, _I(i, j)[1] = 0;//Contrast
		  if (i >= half && j >= half)
			  _I(i, j)[1] = 0, _I(i, j)[2] = 0;; ///R 
		  if (i < half && j >= half)
			  _I(i, j)[0] = 0, _I(i, j)[2] = 0;;// intensity +50	
	  }
	  I = _I;
  }
  return I;
}

int thresh(int val, int thresh){
	if (val > thresh)
		return thresh;
	return val;
}

int thresh(int val){
	/*if (val < 0)
		return val;
	return val;*/
	if (val > 255)
		return 255;
	if (val < 0)
		return 0;
}
static void onMouse(int event, int x, int y, int f, void*){
	cv::Mat image = source.clone();
	cv::Vec3b rgb = image.at<cv::Vec3b>(y, x);
	int b = rgb.val[0];
	int g = rgb.val[1];
	int r = rgb.val[2];

	cv::Mat HSV;
	cv::Mat RGB = image(cv::Rect(x, y, 1, 1));
	cv::cvtColor(RGB, HSV, CV_BGR2HSV);

	cv::Vec3b hsv = HSV.at<cv::Vec3b>(0, 0);
	int h = hsv.val[0];
	int s = hsv.val[1];
	int v = hsv.val[2];

	char name[30];
	sprintf(name, "B=%d", b);
	putText(image, name, cv::Point(150, 40), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "G=%d", g);
	putText(image, name, cv::Point(150, 80), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "R=%d", r);
	putText(image, name, cv::Point(150, 120), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "H=%d", h);
	putText(image, name, cv::Point(25, 40), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "S=%d", s);
	putText(image, name, cv::Point(25, 80), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "V=%d", v);
	putText(image, name, cv::Point(25, 120), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "X=%d", x);
	putText(image, name, cv::Point(25, 300), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(0, 0, 255), 2, 8, false);

	sprintf(name, "Y=%d", y);
	putText(image, name, cv::Point(25, 340), cv::FONT_HERSHEY_SIMPLEX, .7, cv::Scalar(0, 0, 255), 2, 8, false);

	//imwrite("hsv.jpg",image);
	imshow(windowName, image);
}


void loadImgs(){
	images[0].init("lotto-man.jpg");
	images[1].init("lotto-budka.jpeg");
	images[2].init("lotto-twice.jpg");
	images[3].init("not-Lotto.jpg"); 
}

cv::Mat thresholdHue(cv::Mat rgbSrc, int lowerThresh, int upperThresh){
	cv::Mat HSV;
	cv::cvtColor(rgbSrc, HSV, CV_BGR2HSV);
	cv::Mat_<cv::Vec3b> result = HSV;
	for (int i = 0; i < HSV.rows; i++){
		for (int j = 0; j < HSV.cols; j++){
			if (result(i, j)[H] < lowerThresh || result(i, j)[H] > upperThresh)
				result(i, j)[H] = 0;
			else
				result(i, j)[H] = 255;
			result(i, j)[S] = result(i, j)[V] = result(i, j)[H];
		}
	}
	return result;
}


void fillWindow(int i, int j, cv::Mat_<cv::Vec3b> img, int channel, int margin, int SEsize){
	/*window[0][0] = I(i-1, j-1)[channel];
	window[0][1] = I(i - 1, j)[channel];
	window[0][2] = I(i - 1, j + 1)[channel];
	window[1][0] = I(i, j - 1)[channel];
	window[1][1] = I(i, j)[channel];
	window[1][2] = I(i, j + 1)[channel];
	window[2][0] = I(i + 1, j - 1)[channel];
	window[2][1] = I(i + 1, j)[channel];
	window[2][2] = I(i + 1, j + 1)[channel];*/
	for (int a = 0; a < SEsize; a++)
		for (int b = 0; b < SEsize; b++)
			window[a][b] = img(i + a - margin, j + b - margin)[channel];
}

void dilate3(cv::Mat & src, int size){
	cv::Mat_<cv::Vec3b> img = src;
	cv::Mat_<cv::Vec3b> result(src.rows, src.cols);
	int margin = size / 2;
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			if (i - margin < 0 || j - margin < 0 || i + margin >= src.rows || j + margin >= src.cols)
				result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = img(i, j)[0];
			else {
				fillWindow(i, j, img, 0, margin, size);
				bool matched = false;
				for (int a = 0; a < size; a++)
					for (int b = 0; b < size; b++)
						if (window[a][b] == 255 && SE3[a][b] == 1){
							matched = true;
							break;
						}
				if (matched)
					result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = 255;
				else
					result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = img(i, j)[0];
			}
		}
	}
	src = result;
}
void dilate5(cv::Mat & src, int size){
	cv::Mat_<cv::Vec3b> img = src;
	cv::Mat_<cv::Vec3b> result(src.rows, src.cols);
	int margin = size / 2;
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			if (i - margin < 0 || j - margin < 0 || i + margin >= src.rows || j + margin >= src.cols)
				result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = img(i, j)[0];
			else {
				fillWindow(i, j, img, 0, margin, size);
				bool matched = false;
				for (int a = 0; a < size; a++)
					for (int b = 0; b < size; b++)
						if (window[a][b] == 255 && SE5[a][b] == 1){
							matched = true;
							break;
						}
				if (matched)
					result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = 255;
				else
					result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = img(i, j)[0];
			}
		}
	}
	src = result;
}

void dilate7(cv::Mat & src, int size){
	cv::Mat_<cv::Vec3b> img = src;
	cv::Mat_<cv::Vec3b> result(src.rows, src.cols);
	int margin = size / 2;
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			if (i - margin < 0 || j - margin < 0 || i + margin >= src.rows || j + margin >= src.cols)
				result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = img(i, j)[0];
			else {
				fillWindow(i, j, img, 0, margin, size);
				bool matched = false;
				for (int a = 0; a < size; a++)
					for (int b = 0; b < size; b++)
						if (window[a][b] == 255 && SE7[a][b] == 1){
							matched = true;
							break;
						}
				if (matched)
					result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = 255;
				else
					result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = img(i, j)[0];
			}
		}
	}
	src = result;
}

void erode(cv::Mat & src){
	cv::Mat_<cv::Vec3b> img = src;
	cv::Mat_<cv::Vec3b> result(src.rows, src.cols);
	int margin = erodeSEsize / 2;
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			if (i - margin < 0 || j - margin < 0 || i + margin >= src.rows || j + margin >= src.cols)
				result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = img(i, j)[0];
			else {
				fillWindow(i, j, img, 0, margin, erodeSEsize);
				bool matched = true;
				for (int a = 0; a < erodeSEsize; a++)
					for (int b = 0; b < erodeSEsize; b++)
						if (window[a][b] != 255 && SE5[a][b] == 1){
							matched = false;
							break;
						}
				if (matched)
					result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = 255;
				else
					result(i, j)[0] = result(i, j)[1] = result(i, j)[2] = 0;
			}
		}
	}
	src = result;
}
int main(int, char *[]) {
	
    std::cout << "Start ..." << std::endl;
	loadImgs();	
	/*source = originalImgs[2];
	imshow(windowName, source);
	cv::setMouseCallback(windowName, onMouse, 0);*/
	
	//cv::Mat HSV[n];
	//char name[20];
	
	for (int i = 0; i < 3; i++){
		images[i].binaryFromBlue = thresholdHue(images[i].originalImg, 85, 110);
		dilate3(images[i].binaryFromBlue, 3);
		erode(images[i].binaryFromBlue);
		erode(images[i].binaryFromBlue);
		dilate7(images[i].binaryFromBlue, 7);
		dilate3(images[i].binaryFromBlue, 3);
		images[i].findSegments();
		images[i].calcSegmentsParams();
		images[i].filterEllipses();
		images[i].showSegments();
	}

	//images[0].show();

//	findObjects(HSV[0]);
	
	//find objects
	
	//filter out small ones
	
	//look for elipses
	cv::waitKey(-1);
    return 0;
}
