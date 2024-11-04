#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取 PPM 图像
    cv::Mat image = cv::imread("D:\Project2\color - block.ppm", cv::IMREAD_COLOR);

    // 检查图像是否成功读取
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // 显示图像
    cv::imshow("PPM Image", image);
    cv::waitKey(0); // 等待按键输入
    return 0;
}