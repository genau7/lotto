#ifndef SEGMENTFINDER_H
#define SEGMENTFINDER_H
#include "segment.h"

class SegmentFinder{
public:
	
	int** labelPixels(cv::Mat_<cv::Vec3b> img){
		int ** labels = new int*[img.rows];
		for (int j = 0; j < img.cols; ++j)
			labels[j] = new int[img.cols];
		int count = 1;
		for (int i = 0; i < img.rows; ++i)
			for (int j = 0; j < img.cols; ++j) {
				if (img(i, j)[0] != 0)
					labels[i][j] = count++;
				else
					labels[i][j] = 0;
			}
	}

	void indexSegments(int** labels, int rows, int cols){
		bool segmentationDone = false;
		while (!segmentationDone){
			bool notFinished1 = topDownPass(labels, rows, cols);
			bool notFinished2 = bottomUpPass(labels, rows, cols);
			segmentationDone = !(notFinished1 || notFinished2);
		}
	}

private:
	bool bottomUpPass(int** labels, int rows, int cols){
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
	bool topDownPass(int ** labels, int rows, int cols){
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
};

#endif //SEGMENTFINDER_H