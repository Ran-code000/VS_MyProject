#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // ��ȡ PPM ͼ��
    cv::Mat image = cv::imread("D:\Project2\color - block.ppm", cv::IMREAD_COLOR);

    // ���ͼ���Ƿ�ɹ���ȡ
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // ��ʾͼ��
    cv::imshow("PPM Image", image);
    cv::waitKey(0); // �ȴ���������
    return 0;
}