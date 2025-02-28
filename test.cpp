#include <iostream>
#include<Eigen/Dense>
#include <vector>
#include <iomanip>

int main() {
    // 创建一个3x10x10的张量，填充1到300的值
    const int depth = 3;
    const int height = 10;
    const int width = 10;
    const int total_size = depth * height * width;
    
    // 创建一维数组来存储所有数据
    std::vector<float> tensor_data(total_size);
    
    // 填充1到300的值
    for (int i = 0; i < total_size; ++i) {
        tensor_data[i] = i + 1;
    }
    // 为了更好地理解内存排布，也打印原始数据的3D结构
    std::cout << "\nOriginal 3D Tensor (3x10x10):" << std::endl;
    for (int d = 0; d < depth; ++d) {
        std::cout << "Channel " << d << ":" << std::endl;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int index = d * (height * width) + h * width + w;
                std::cout << std::setw(4) << tensor_data[index];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    


    // 使用Eigen::Map将数据映射为矩阵
    // 注意：这里将3x10x10的数据映射为100x3的矩阵
    Eigen::Map<Eigen::MatrixXf> img_seg_(tensor_data.data(), 100, 3);
    
    // 打印映射后的矩阵
    std::cout << "Mapped Matrix (100x3):" << std::endl;
    std::cout << std::fixed << std::setprecision(0); // 设置打印格式
    std::cout << img_seg_ << std::endl;
    
    
    return 0;
}