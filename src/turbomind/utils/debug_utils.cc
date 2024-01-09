#include "src/turbomind/utils/debug_utils.h"

std::string getFilePath(const std::string& filename) {
    std::string prefix = "/data/yocto_bak/analyse/lmdeploy/";
    return prefix + filename + ".bin";
}

void saveDataEuropa(int num_element, half* d_data, const std::string& file_) {
    std::string filePath = getFilePath(file_);
    // Allocate host memorys
    half* h_data = new half[num_element];

    // Copy data from device to host
    cudaMemcpy(h_data, d_data, num_element * sizeof(half), cudaMemcpyDeviceToHost);
    // Convert data from half to float
    std::vector<float> vec_float(num_element);
    for (int i = 0; i < num_element; ++i) {
        vec_float[i] = static_cast<float>(h_data[i]);
        // printf("%f\n", vec_float[i]);
    }
    // Save data to file
    std::ofstream outfile(filePath, std::ios::binary);
    if (outfile.is_open()) {
        printf("\n dumping to file: %s \n", filePath.c_str());
        outfile.write((char*)vec_float.data(), num_element * sizeof(float));
        if (outfile.fail()) {
            printf("Error writing to file.\n");
        }
        outfile.close();
    } else {
        printf("Unable to open file.\n");
    }
    // Free host memory
    delete[] h_data;
}