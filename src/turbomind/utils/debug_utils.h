#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_fp16.h> 

// 函数声明
extern std::string getFilePath(const std::string& filename);

extern void saveDataEuropa(int num_element, half* d_data, const std::string& file_);

#endif // DEBUG_UTILS_H