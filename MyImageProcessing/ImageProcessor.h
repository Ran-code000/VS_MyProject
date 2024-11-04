#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <tuple>
#include <string>

class ImageProcessor {
public:
    cv::Mat readImage(const std::string& path);
    void saveImage(const std::string& path, const cv::Mat& img);
    void displayImage(const std::string& windowName, const cv::Mat& img);

    std::vector<std::tuple<int, int, int>> compressImage(const cv::Mat& img);
    void saveCompressedData(const std::string& path, const std::vector<std::tuple<int, int, int>>& data);
    cv::Mat decompressImage(const std::string& path);

    cv::Mat convertToGrayscale(const cv::Mat& img, int type = 0);
    cv::Mat convertToColorscale(const cv::Mat& img);

    cv::Mat resizeImage(const cv::Mat& img, const cv::Size& size);
};
