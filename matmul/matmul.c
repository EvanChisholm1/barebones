#include<stdio.h>
#include<stdlib.h>

#define mf struct Matrixf

struct Matrixf {
    float *a;
    int width;
    int height;
};

mf createMatrixf(int width, int height) {
    mf out;
    out.width = width;
    out.height = height;
    out.a = (float *)malloc(sizeof(float) * width * height);
    return out;
}

mf matmul(struct Matrixf a, struct Matrixf b) {
    mf out = createMatrixf(b.width, a.height);

    for(int col = 0; col < out.width; col++) {
        for(int row = 0; row < out.height; row++) {
            float item = 0;

            for(int j = 0; j < a.width; j++) {
                item += a.a[row * a.width + j] * b.a[j * b.width + col];
            }

            out.a[row * out.width + col] = item;
        }
    }

    return out;
}

int main() {
    FILE *f;
    f = fopen("mats.bin", "rb");

    float *buffer = (float *)malloc(sizeof(float) * (2 * 4 + 4));
    fread(buffer, sizeof(float), (2 * 4 + 4), f);

    mf a;
    a.a = buffer;
    a.width = 4;
    a.height = 2;

    mf b = createMatrixf(1, 4);
    b.a[0] = 1.0;
    b.a[1] = 1.0;
    b.a[2] = 1.0;
    b.a[3] = 1.0;

    mf c = matmul(a, b);

    printf("Result:\n");
    for(int i = 0; i < c.height; i++) {
        for(int j = 0; j < c.width; j++) {
            printf("%f ", c.a[i * c.width + j]);
        }
        printf("\n");
    }
    return 0;
}
