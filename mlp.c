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
mf fc1_weight = {NULL, 0, 0};
vf fc1_bias = {NULL, 0};
vf fc2_weight = {NULL, 0};
vf fc2_bias = {NULL, 0};

void load_params() {
    FILE *file;

    file = fopen("weights.bin", "rb");
    if(file == NULL) {
        printf("Error: can't open file\n");
        exit(1);
    }

    fread(&config, sizeof(config), 1, file);

    float *buffer = (float *)malloc(sizeof(float) * config.n_params);
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
    fc1_weight.width = config.hidden_size;
    fc1_weight.height = config.block_size * config.embed_size;

    // printf("fc1 weight:\n");
    // print_mf(fc1_weight);

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


mf create_mf(int width, int height) {
    mf out;
    out.width = width;
    out.height = height;
    out.a = (float *)malloc(sizeof(float) * width * height);
    return out;
}

mf matmul(mf a, mf b) {
    mf out = create_mf(b.width, a.height);

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

vf add_bias(vf a, vf b) {
    for(int i = 0; i < b.size; i++) {
        a.a[i] += b.a[i];
    }

    return a;
}

vf as_vf(mf m) {
    vf ret;
    ret.a = m.a;
    ret.size = m.width * m.height;
    return ret;
}

mf as_mf(vf v, int width, int height) {
    mf ret;
    ret.a = v.a;
    ret.width = width;
    ret.height = height;
    return ret;
}

float max(float a, float b) {
    if(a > b) return a;
    else return b;
}

void relu(float *a, int size) {
    for(int i = 0; i < size; i++) {
        a[i] = max(0, a[i]);
    }
}

void print_vf(vf v) {
    for(int i = 0; i < v.size; i++) {
        printf("%f ", v.a[i]);
    }
    printf("\n");
}

void print_mf(mf m) {
    for(int i = 0; i < m.height; i++) {
        printf("|");
        for(int j = 0; j < m.width; j++) {
            printf("%f ", m.a[i * m.width + j]);
        }
        printf("|\n");
    }
}

vf embed(char *str) {
    vi int_index = string_to_int(str);

    float *embed = (float *)malloc(sizeof(float) * config.embed_size * int_index.size);

    for(int i = 0; i < int_index.size; i++) {
        for(int j = 0; j < config.embed_size; j++) {
            embed[i * config.embed_size + j] = embed_weight.a[int_index.a[i] * config.embed_size + j];
        }
    }

    return (vf){embed, config.embed_size * int_index.size};
}

vf relu_v(vf v) {
    relu(v.a, v.size);
    return v;
}

mf relu_mf(mf m) {
    relu_v(as_vf(m));

    return m;
}

int main() {
    load_params();
    printf("hidden size: %d\n", config.hidden_size);

    vf embedding = embed("hell");
    print_vf(embedding);
    mf e = as_mf(embedding, 1, config.embed_size * config.block_size);

    printf("E shape: %d, %d\n", e.width, e.height);
    printf("FC1 weight shape: %d, %d\n", fc1_weight.width, fc1_weight.height);

    printf("\n\nfc1\n");
    vf fc1 = relu_v(add_bias(as_vf(matmul(fc1_weight, e)), fc1_bias));

    print_vf(fc1);

    free(embedding.a);
    free(params.a);
    free(fc1.a);

    return 0;
}
