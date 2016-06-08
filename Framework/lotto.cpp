#include "segmentFinder.h"




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




struct SegmentPair{
	Segment* s1;
	Segment* s2;
	float distance;
	float areaRatio;
	LinearFunction axis;
	Pixel upperLeft;
	Pixel bottomRight;
	float axisAngle;
	bool isHorizontal;
	float averageD; //diameter
	int maxCols, maxRows;
	SegmentPair(Segment* first, Segment* second, int maxCols, int maxRows){
		if (first->minCol <= second->minCol){
			s1 = first;
			s2 = second;
		}
		else{
			s1 = second;
			s2 = first;
		}
		Pixel center1(s1->massCenter.x + s1->minCol, s1->massCenter.y + s1->minRow);
		Pixel center2(s2->massCenter.x + s2->minCol, s2->massCenter.y + s2->minRow);
		float rowDistance = center1.y - center2.y;
		float colDistance = center1.x - center2.x;
		distance = sqrt(rowDistance*rowDistance + colDistance*colDistance);
		areaRatio = fabs(s2->area * 1.0 / s1->area);
		axis.calcCoefficients(center1, center2);
		upperLeft.x = (s1->minCol < s2->minCol) ? s1->minCol : s2->minCol;
		upperLeft.y = (s1->minRow < s2->minRow) ? s1->minRow : s2->minRow;
		bottomRight.x = (s1->maxCol > s2->maxCol) ? s1->maxCol : s2->maxCol;
		bottomRight.y = (s1->maxRow > s2->maxRow) ? s1->maxRow : s2->maxRow;
		axisAngle = 180.0 / 3.14 * fabs(axis.a);
		isHorizontal = ((int)axisAngle / 45 == 0) ? true: false;
		averageD = (s1->cols + s2->cols) * 1.0 / 2;
		this->maxCols = maxCols;
		this->maxRows = maxRows;
	}
	bool pixelOnAxis(int row, int col){
		return axis.onTheLine(row, col);
	}
	void drawCenters(cv::Mat_<uchar>& img){
		Pixel a = s1->massCenter;
		Pixel b = s2->massCenter;
		img[s1->massCenter.y + s1->minRow][s1->massCenter.x + s1->minCol] = 255;
		img[s2->massCenter.y + s2->minRow][s2->massCenter.x +s2->minCol] = 255;
	}
	void expandBoundingBox(){
		float upperLeftDec, bottomRightInc;
		float dxUpperLeft, dyUpperLeft, dxBottomRight, dyBottomRight;
		if (s1->minRow <= s2->minRow){
			upperLeftDec = -0.5 * sqrt(s1->cols * s1->cols + s1->rows * s1->rows);
			bottomRightInc = 0.5 * sqrt(s2->cols * s2->cols + s2->rows * s2->rows);
		}
		else{
			upperLeftDec = -0.5 * sqrt(s2->cols * s2->cols + s2->rows * s2->rows);
			bottomRightInc = 0.5 * sqrt(s1->cols * s1->cols + s1->rows * s1->rows);
		}
		dxUpperLeft = upperLeftDec / sqrt(axis.a*axis.a + 1);
		dyUpperLeft = axis.a * dxUpperLeft;
		dxBottomRight = bottomRightInc / sqrt(axis.a*axis.a + 1);
		dyBottomRight = axis.a * dxBottomRight;
		upperLeft.x = (upperLeft.x + dxUpperLeft >= 0) ? upperLeft.x + dxUpperLeft : 0;
		upperLeft.y = (upperLeft.y + dyUpperLeft >= 0) ? upperLeft.y + dyUpperLeft : 0;
		bottomRight.x = (bottomRight.x + dxBottomRight >= 0) ? bottomRight.x + dxBottomRight : 0;
		bottomRight.y = (bottomRight.y + dyBottomRight >= 0) ? bottomRight.y + dyBottomRight : 0;
	}
};

typedef std::vector<std::vector<SegmentPair>> LottoWritings;

struct Img {
	cv::Mat originalImg;
	cv::Mat_<cv::Vec3b> binaryFromBlue;
	cv::Mat segmented; 
	int** labels;
	std::string name;
	int rows, cols;
	std::vector<Segment> segments;
	std::vector<SegmentPair> segmentPairs;

	void init(std::string filename){
		originalImg = cv::imread(filename.c_str());
		name = filename;
		setup();
	}

	void init(cv::Mat img_in, std::string filename = ""){
		name = filename;
		originalImg = img_in;
		setup();
	}

	void setup(){
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
			sprintf(m7, "%f", s.width2HeightRatio);
			//sprintf(m7, "%d", s.area);
			putText(segmented, m7, cv::Point(s.minCol, s.minRow), cv::FONT_HERSHEY_SIMPLEX, .4, cv::Scalar(255, 200, 255), 1, 8, false);
		}
		cv::imshow(name.c_str(), segmented);
	}
	void filterEllipses(){
		for (int k = 0; k < segments.size();){
			Segment s = segments.at(k);
			if (s.M7 > 0.0071 || s.M7 < 0.00638 || s.width2HeightRatio < 0.68 ||
			          s.width2HeightRatio > 1.063 || s.area < 150){
				s.clear();
				segments.erase(segments.begin() + k);
			}
			else
				++k;
		}
	}

	bool findSegmentPairs(){
		for (int a = 0; a < segments.size() - 1; ++a)
			for (int b = a + 1; b < segments.size(); ++b){
				SegmentPair pair(&segments.at(a), &segments.at(b), cols, rows);
				float distanceCoeff = pair.distance / pair.averageD;
				if (pair.areaRatio > 0.52  && (distanceCoeff >2.3 && distanceCoeff < 2.5)){
					pair.expandBoundingBox();
					segmentPairs.push_back(pair);				
				}
			}
		return (segmentPairs.size() > 0);
	}
	void colorSegments(){
		cv::Mat_<uchar> temp = segmented;
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j) {
				temp(i, j) = (labels[i][j] == 0) ? 0 : 100;
				//temp(i, j) = (labels[i][j] * 10) % 255;
			}

		//draw axis
		for (int k = 0; k < segmentPairs.size(); ++k){
			SegmentPair pair = segmentPairs.at(k);
			for (int i = 0; i < rows; ++i)
				for (int j = 0; j < cols; ++j) {
					if (pair.pixelOnAxis(i, j))
						temp[i][j] = 255;
				}
		}

		//draw bounding box for each segment pair found
		for (int k = 0; k < segmentPairs.size(); ++k){
			SegmentPair pair = segmentPairs.at(k);
			pair.drawCenters(temp);
			for (int i = pair.upperLeft.y; i <= pair.bottomRight.y; ++i)
				if (i == pair.upperLeft.y || i == pair.bottomRight.y)
					for (int j = pair.upperLeft.x; j <= pair.bottomRight.x; ++j) 
					//if (pair.pixelOnAxis(i, j))
						temp[i][j] = 255;

			for (int i = pair.upperLeft.y; i <= pair.bottomRight.y; ++i)
				for (int j = pair.upperLeft.x; j <= pair.bottomRight.x; ++j)
					if (j == pair.upperLeft.x || j == pair.bottomRight.x)
						temp[i][j] = 255;
			
					
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

	void findLottoInSegmentsPairs(){
		binaryFromBlue.release();
		segmented.release();
		for (int k = 0; k < segmentPairs.size(); ++k){
			cv::Mat temp;
			char pairName[30];
			sprintf(pairName, "%s_Lotto%d", name.c_str(), k);
			originalImg.copyTo(temp);
			SegmentPair pair = segmentPairs.at(k);
			cv::Rect ROI(pair.upperLeft.x, pair.upperLeft.y, pair.bottomRight.x - pair.upperLeft.x, pair.bottomRight.y - pair.upperLeft.y);
			cv::Mat boundedMat = temp(ROI);
			//Img boundedImg;
		//	boundedImg.init(boundedMat, pairName);
			cv::Mat_<cv::Vec3b> justBlack = thresholdValue(boundedMat, 0, 65);
			dilate3(justBlack, 3);
			erode(justBlack);
			//erode(justBlack);
			//dilate7(justBlack, 7);
			dilate3(justBlack, 3);
			cv::imshow(pairName, justBlack);

		}
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
	/*void drawCenters(){
		cv::Mat_<cv::Vec3b> tempImg = boundedImg;
		tempImg(center.y, center.x)[1] = 255;
		tempImg(box.center.y, box.center.x)[2] = 180;
		boundedImg = tempImg;
	}*/
	

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





int main(int, char *[]) {
	
    std::cout << "Start ..." << std::endl;
	loadImgs();	
	/*LottoWritings lottoWritings;
	source = images[2].originalImg;
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
		images[i].findSegmentPairs();

		//images[i].findLottoInSegmentsPairs();
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
