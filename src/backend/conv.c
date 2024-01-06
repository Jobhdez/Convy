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

float weight[1][1][3][3] = {{{{0.05152066797018051, 0.24578683078289032, 0.046719830483198166}, {-0.00031745433807373047, -0.11531953513622284, -0.12301242351531982}, {-0.14917227625846863, 0.03384590148925781, 0.12455253303050995}}}};

float bias = -0.2332494705915451;
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