#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main() {
    FILE *file;
    float buffer[8 * 26 + 10];

    file = fopen("weights.bin", "rb");
    if(file == NULL) {
        printf("Error: can't open file\n");
        exit(1);
    }

    size_t elements_read = fread(buffer, sizeof(float), 8 * 26 + 10, file);

    if(elements_read == 0) {
        perror("Error: can't read file\n");
        fclose(file);
        exit(1);
    }

    for(size_t i = 0; i < elements_read; i++) {
        printf("%f\n", buffer[i]);
    }
    fclose(file);
    return 0;
}
