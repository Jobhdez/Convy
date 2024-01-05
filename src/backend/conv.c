#include "runtime.c"
#include <stdio.h>


int main() {

int batch_size = 1;

int channels = 1;

int input_height = 3;

int input_width = 3;

int filter_height = 3;

int filter_width = 3;

float input_data[1][1][3][3] = {{{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}}}};

float weight[1][1][3][3] = {{{{0.07503136247396469, 0.25153809785842896, -0.0697178840637207}, {-0.13205358386039734, -0.14122705161571503, 0.18474909663200378}, {-0.08937716484069824, -0.27898573875427246, -0.0021663110237568617}}}};

float bias = -0.17763817310333252;
float output[1][1][3][3];

 convolution(input_data, weight, &bias, batch_size, channels, input_height, input_width, filter_height, filter_width, output);



    // Print the result
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < 1; ++j) {  // Assuming output has only one channel
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    printf("%f ", output[i][j][k][l]);
                }
                
            }
        }
    }return 0;
}