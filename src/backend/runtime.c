#include <stdio.h>

void convolution(float input_data[][1][3][3], float weight[][1][3][3], float* bias, int batch_size, int channels, int input_height, int input_width, int filter_height, int filter_width, float output[][1][3][3]) {
    int output_height = input_height - filter_height + 1;
    int output_width = input_width - filter_width + 1;

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < 1; ++c) {  // Assuming output has only one channel
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    float sum = 0.0;

                    for (int i = 0; i < filter_height; ++i) {
                        for (int j = 0; j < filter_width; ++j) {
                            sum += input_data[b][c][h + i][w + j] * weight[0][0][i][j];
                        }
                    }

                    output[b][c][h][w] = sum + (*bias);
                }
            }
        }
    }
}

/*
int main() {
    // Example usage
    int batch_size = 1;
    int channels = 1;
    int input_height = 5;
    int input_width = 5;
    int filter_height = 3;
    int filter_width = 3;

    float input_data[1][1][5][5] = {{{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}}}};
    float weight[1][1][3][3] = {{{{0, 1, 0}, {1, 2, 1}, {0, 1, 0}}}};
    float bias = 1.0;
    float output[1][1][3][3];

    convolution(input_data, weight, &bias, batch_size, channels, input_height, input_width, filter_height, filter_width, output);

    // Print the result
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < 1; ++j) {  // Assuming output has only one channel
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    printf("%f ", output[i][j][k][l]);
                }
                printf("\n");
            }
        }
    }

    return 0;
}
*/
