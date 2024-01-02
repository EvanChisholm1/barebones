#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

#define vi struct Veci
#define vf struct Vecf
#define mf struct Matf

char vocab[27] = ".abcdefghijklmnopqrstuvwxyz";

struct Veci {
    int *a;
    int size;
};

struct Vecf {
    float *a;
    int size;
};

struct Matf {
    float *a;
    int width;
    int height;
};

struct MlpConfig {
    int embed_size;
    int hidden_size;
    int vocab_size;
    int block_size;
    int output_size;
    int n_params;
};

vi string_to_int(char *str) {
    int *index= (int *)malloc(sizeof(int) * strlen(str));
    for(int i = 0; i < strlen(str); i++) {
        for(int j = 0; j < 27; j++) {
            if(str[i] == vocab[j]) {
                index[i] = j;
                break;
            }
        }
    }

    vi out;
    out.a = index;
    out.size = strlen(str);

    return out;
}

char *int_to_string(vi int_index) {
    char *str = (char *)malloc(sizeof(char) * (int_index.size + 1));

    for(int i = 0; i < int_index.size; i++) {
        printf("%d, %c\n", int_index.a[i], vocab[int_index.a[i]]);
        sprintf(str + strlen(str), "%c", vocab[int_index.a[i]]);
    }

    return str;
}

struct MlpConfig config;
vf params = {NULL, 0};
vf embed_weight = {NULL, 0};
vf fc1_weight = {NULL, 0};
vf fc1_bias = {NULL, 0};
vf fc2_weight = {NULL, 0};
vf fc2_bias = {NULL, 0};

void load_params() {
    FILE *file;
    float *buffer = (float *)malloc(sizeof(float) * config.n_params);

    file = fopen("weights.bin", "rb");
    if(file == NULL) {
        printf("Error: can't open file\n");
        exit(1);
    }

    size_t elements_read = fread(buffer, sizeof(float), config.n_params, file);

    if(elements_read == 0) {
        perror("Error: can't read file\n");
        fclose(file);
        exit(1);
    }

    params.a = buffer;
    params.size = config.n_params;

    int embed_weight_size = config.embed_size * config.vocab_size;
    int embed_offset = 0;
    embed_weight.a = params.a + embed_offset;
    embed_weight_size = embed_weight_size;

    int fc1_weight_size = config.hidden_size * config.block_size * config.embed_size;
    int fc1_weight_offset = embed_weight_size;
    fc1_weight.a = params.a + fc1_weight_offset;
    fc1_weight.size = fc1_weight_size;

    int fc1_bias_size = config.hidden_size;
    int fc1_bias_offset = fc1_weight_offset + fc1_weight_size;
    fc1_bias.a = params.a + fc1_bias_offset;
    fc1_bias.size = fc1_bias_size;

    int fc2_weight_size = config.hidden_size * config.output_size;
    int fc2_weight_offset = fc1_bias_offset + fc1_bias_size;
    fc2_weight.a = params.a + fc2_weight_offset;
    fc2_weight.size = fc2_weight_size;

    int fc2_bias_size = config.output_size;
    int fc2_bias_offset = fc2_weight_offset + fc2_weight_size;
    fc2_bias.a = params.a + fc2_bias_offset;
    fc2_bias.size = fc2_bias_size;

    fclose(file);
}

void print_vf(vf v) {
    for(int i = 0; i < v.size; i++) {
        printf("%f ", v.a[i]);
    }
    printf("\n");
}

int main() {
    config.block_size = 4;
    config.embed_size = 8;
    config.hidden_size = 128;
    config.vocab_size = 27;
    config.output_size = 27;
    config.n_params = 7923;

    load_params();

    // print each param
    // for(int i = 0; i < config.n_params; i++) {
    //     printf("%f\n", params.a[i]);
    // }

    print_vf(fc2_bias);

    free(params.a);

    return 0;
}
