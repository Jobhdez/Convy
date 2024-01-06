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

float weight[1][1][3][3] = {{{{0.113400898873806, 0.28415676951408386, -0.16677796840667725}, {0.022547762840986252, 0.24742528796195984, 0.010448455810546875}, {-0.1895950734615326, -0.05162879079580307, -0.09433500468730927}}}};

float bias = -0.014137348160147667;
float output[1][1][3][3];

convolution(input_data, weight, &bias, batch_size, channels, input_height, input_width, filter_height, filter_width, output);

for (int i = 0; i < batch_size; ++i) {
  for (int j = 0; j < 1; ++j) {
     for (int k = 0; k < 3; ++k) {
       for (int l = 0; l < 3; ++l) {
         printf("%f ", output[i][j][k][l]);
       }
     }
  }
}
return 0;
}