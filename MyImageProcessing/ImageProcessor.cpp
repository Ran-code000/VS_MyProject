#include "ImageProcessor.h"
#include <fstream>

cv::Mat ImageProcessor::readImage(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
    }
    return img;
}

void ImageProcessor::saveImage(const std::string& path, const cv::Mat& img) {
    cv::imwrite(path, img);
}

void ImageProcessor::displayImage(const std::string& windowName, const cv::Mat& img) {
    if (img.empty()) {
        std::cerr << "Error: Image is empty!" << std::endl;
        return;
    }
    cv::imshow(windowName, img);
    cv::waitKey(0);
}

std::vector<std::tuple<int, int, int>> ImageProcessor::compressImage(const cv::Mat& img) {
    std::vector<std::tuple<int, int, int>> compressedData;
    compressedData.emplace_back(-1, img.rows, img.cols);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            compressedData.emplace_back(pixel[0], pixel[1], pixel[2]);
        }
    }
    return compressedData;
}

void ImageProcessor::saveCompressedData(const std::string& path, const std::vector<std::tuple<int, int, int>>& data) {
    std::ofstream ofs(path, std::ios::binary);
    for (const auto& triplet : data) {
        ofs.write(reinterpret_cast<const char*>(&std::get<0>(triplet)), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&std::get<1>(triplet)), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&std::get<2>(triplet)), sizeof(int));
    }
    ofs.close();
}

cv::Mat ImageProcessor::decompressImage(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    std::vector<std::tuple<int, int, int>> data;
    while (ifs) {
        int r, g, b;
        ifs.read(reinterpret_cast<char*>(&r), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&g), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&b), sizeof(int));
        if (ifs) {
            data.emplace_back(r, g, b);
        }
    }
    ifs.close();

    if (data.empty()) {
        std::cerr << "No data found!" << std::endl;
        return cv::Mat();
    }

    int rows = std::get<1>(data[0]);
    int cols = std::get<2>(data[0]);

    cv::Mat img(rows, cols, CV_8UC3);
    int idx = 1;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (idx < data.size()) {
                img.at<cv::Vec3b>(i, j) = cv::Vec3b(std::get<0>(data[idx]), std::get<1>(data[idx]), std::get<2>(data[idx]));
                ++idx;
            }
        }
    }

    return img;
}

cv::Mat ImageProcessor::convertToGrayscale(const cv::Mat& img, int type) {
    cv::Mat grayImg;

    if (img.empty()) {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return grayImg;
    }
    /*
        浮点算法：Gray=0.299R+0.587G+0.114B

        整数方法：Gray=(R30+G59+B*11)/100

        移位方法：Gray=(R28+G151+B*77)>>8

        平均值法：Gray=（R+G+B）/3

        最大值法：Gray = max(R,G,B)

        最小值法：Gray = min(R,G,B)

        仅取绿色：Gray=G
    */
    switch (type) {
    case 0:
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
        break;
    case 1: {
        std::cout << "float algorithm\n";
        grayImg = cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(0.0));
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
                float gray = 0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2];
                grayImg.at<float>(i, j) = gray;
            }
        }
        break;
    }
    case 2: {
        std::cout << "int algorithm\n";
        grayImg = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
                int gray = (30 * pixel[0] + 59 * pixel[1] + 11 * pixel[2]) / 100;
                grayImg.at<uchar>(i, j) = static_cast<uchar>(gray);
            }
        }
        break;
    }
    case 3: {
        std::cout << "shift algorithm\n";
        grayImg = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
                int gray = (28 * pixel[0] + 151 * pixel[1] + 77 * pixel[2]) >> 8;
                grayImg.at<uchar>(i, j) = static_cast<uchar>(gray);
            }
        }
        break;
    }
    case 4: {
        std::cout << "average algorithm\n";
        grayImg = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
                int gray = (pixel[0] + pixel[1] + pixel[2]) / 3;
                grayImg.at<uchar>(i, j) = static_cast<uchar>(gray);
            }
        }
        break;
    }
    case 5: {
        std::cout << "max algorithm\n";
        grayImg = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
                int gray = std::max({ pixel[0], pixel[1], pixel[2] });
                grayImg.at<uchar>(i, j) = static_cast<uchar>(gray);
            }
        }
        break;
    }
    case 6: {
        std::cout << "min algorithm\n";
        grayImg = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
                int gray = std::min({ pixel[0], pixel[1], pixel[2] });
                grayImg.at<uchar>(i, j) = static_cast<uchar>(gray);
            }
        }
        break;
    }
    default:
        std::cerr << "Invalid type! Please input a number from 0 to 6." << std::endl;
        return grayImg;
    }

    return grayImg;
}

cv::Mat loadPtsInHull(const std::string& npyFile) {
    // Use numpy to load the npy file into a Mat
    cv::Mat ptsInHull;
    std::ifstream input(npyFile, std::ios::binary);
    if (input.is_open()) {
        // Read the npy header
        char header[256];
        input.read(header, 256);

        // Read the data
        int numPts = 313, numDims = 2;
        ptsInHull.create(numPts, numDims, CV_32F);
        input.read(reinterpret_cast<char*>(ptsInHull.data), numPts * numDims * sizeof(float));
        input.close();
    }
    return ptsInHull;
}

cv::Mat ImageProcessor::convertToColorscale(const cv::Mat& img) {
    if (img.empty()) {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return cv::Mat(); // 返回空的 Mat
    }
    // 加载深度学习模型
    std::string protoFile = "D:\\Deep learning model\\colorization-caffe\\models\\colorization_deploy_v2.prototxt";
    std::string weightsFile = "D:\\Deep learning model\\colorization-caffe\\models\\dummy.caffemodel";
    std::string ptsFile = "D:\\Deep learning model\\colorization - caffe\\resources\\pts_in_hull.npy";

    cv::dnn::Net net = cv::dnn::readNetFromCaffe(protoFile, weightsFile);

    // 读取类中心点
    cv::Mat ptsInHull = loadPtsInHull(ptsFile);

    // 调整网络的最后一层
    cv::Mat pts(1, ptsInHull.total(), CV_32F, ptsInHull.ptr<float>());
    cv::Mat class8_ab(1, ptsInHull.total(), CV_32F, ptsInHull.ptr<float>());

    // Create a blob from the data and set it to the network
    std::vector<cv::Mat> blobs;
    blobs.push_back(cv::Mat(1, 1, CV_32F, cv::Scalar(0)));
    blobs.push_back(class8_ab);

    net.getLayer(net.getLayerId("class8_ab"))->blobs = blobs;

    // 预处理图像
    cv::Mat labImage, l;
    cv::Mat grayImage;
    if (img.channels() == 1) {
        grayImage = img.clone(); // 如果是灰度图，直接使用
    }
    else {
        cv::cvtColor(img, grayImage, cv::COLOR_RGB2GRAY);
    }
    cv::cvtColor(grayImage, labImage, cv::COLOR_GRAY2RGB);
    labImage.convertTo(labImage, CV_32F, 1.0 / 255);
    cv::cvtColor(labImage, labImage, cv::COLOR_RGB2Lab);
    std::vector<cv::Mat> labPlanes(3);
    cv::split(labImage, labPlanes);
    l = labPlanes[0];
    l -= 50;

    // 前向传播
    cv::Mat inputBlob = cv::dnn::blobFromImage(l);
    net.setInput(inputBlob);
    cv::Mat ab = net.forward();

    // 后处理
    cv::Size imgSize = img.size();
    cv::resize(ab, ab, imgSize);
    std::vector<cv::Mat> labPlanesOut(3);
    labPlanesOut[0] = l + 50;
    cv::split(ab, labPlanesOut);
    cv::merge(labPlanesOut, labImage);
    cv::cvtColor(labImage, labImage, cv::COLOR_Lab2BGR);
    labImage.convertTo(labImage, CV_8U, 255);

    return labImage;


}

cv::Mat ImageProcessor::resizeImage(const cv::Mat& img, const cv::Size& size) {
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, size);
    return resizedImg;
}
