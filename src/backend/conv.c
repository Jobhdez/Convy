
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
    float weight[1][1][3][3] = {{{{0.1206025704741478, 0.2804994285106659, 0.10350386798381805}, {-0.31444603204727173, -0.3109368085861206, -0.22286033630371094}, {-0.12215566635131836, 0.2765805721282959, 0.25621065497398376}}}};
    float bias = -0.24808375537395477;
    float output[1][1][3][3];

    convolution(input_data, weight, &bias, batch_size, channels, input_height, input_width, filter_height, filter_width, output);

    printf("%f", output[0][0][0][0]);
       
    return 0;
}
