#include <iostream>
#include "ImageProcessor.h"

int main() {
    ImageProcessor ip;
     

    std::cout << "Start reading image......\n";
    cv::Mat img = ip.readImage("D:\\Project2\\1\\lena-512-gray.png");
    ip.displayImage("Original Image", img);
    std::cout << "After reading image......\n";


    std::cout << "Start compressing image......\n";
    std::vector<std::tuple<int, int, int>> compressedData = ip.compressImage(img);
    ip.saveCompressedData("D:\\Project2\\1\\compressed_image.dat", compressedData);
    std::cout << "After compressing image......\n";


    std::cout << "Start depressing image......\n";
    cv::Mat decompressedImg = ip.decompressImage("compressed_image.dat");
    ip.saveImage("D:\\Project2\\1\\decompressed_image.ppm", decompressedImg);
    std::cout << "After depressing image......\n";


    std::cout << "Start converting color to gray.......\n";
    cv::Mat _img = ip.readImage("D:\\Project2\\1\\color-block.png");
    ip.displayImage("Color Image", _img);

    std::cout << "8 ways:\n";
    for (int i = 0; i <= 6; i++){
        std::cout << "way" << i << std::endl;
        cv::Mat grayImg = ip.convertToGrayscale(_img, i);
        ip.saveImage("D:\\Project2\\1\\gray_image.ppm", grayImg);
        ip.displayImage("Grayscale Image", grayImg);
    }
   
    std::cout << "After converting color to gray.......\n";


    std::cout << "Start converting gray to color.......\n";
    ip.displayImage("Gray image", img);
    cv::Mat colorImg = ip.convertToColorscale(img);
    ip.displayImage("Color Image", colorImg);


    std::cout << "Start resizing image\n";
    while (1) {
        int rows, cols;
        std::cout << "Please input the size that you want to resize(format : rows * cols)\n";
        try {
            std::cin >> rows >> cols;
        }
        catch (const std::exception& e){
            std::cerr << "error: " << e.what() << std::endl;
        }
        cv::Mat resizedImg = ip.resizeImage(img, cv::Size(rows, cols));
        ip.displayImage("Original Image", img);
        ip.saveImage("D:\\Project2\\1\\resized_image.ppm", resizedImg);
        ip.displayImage("Resized Image", resizedImg);
        std::cout << "Do you want to jump out the resized_Image processing? please input y(means yes) or n(means no)\n";
        char ch;
        std::cin >> ch;
        ch = std::tolower(ch);
        if (ch == 'y') break;
        else if (ch == 'n') continue;
        else {
            std::cerr << "Invaild input : please input y(means yes) or n(means no)\n";
        }
    }
    std::cout << "After resizing image\n";
    std::cout << "Win!!!!!\n";
    return 0;
}