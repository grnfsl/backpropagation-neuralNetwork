#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <time.h>
#include <math.h>


//genrate random double
double rand_from(double min, double max)
{
    double range = max - min;
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

Mat* create_mat(unsigned int n, unsigned int m)
{
    double* arr = (double*)malloc(n * m * sizeof (double));
    Mat* mat = (Mat*)malloc(sizeof (Mat));
    mat->n = n;
    mat->m = m;
    mat->mat = arr;
    return mat;
}

Mat* create_random_mat(unsigned int n, unsigned int m, double min, double max)
{
    srand(time(NULL));
    Mat* mat = create_mat(n, m);
    unsigned int len = n*m, i;

    for(i = 0; i < len; ++i){
        mat->mat[i] = rand_from(min, max);
    }

    return mat;
}

double* create_vector(unsigned int n)
{
    double* x = (double*)malloc(n * sizeof (double));
    return x;
}

double* create_random_vector(unsigned int n, double min, double max)
{
    srand(time(NULL));
    double* x = (double*)malloc(n * sizeof (double));
    unsigned i;
    for(i = 0; i < n; ++i)
        x[i] = rand_from(min, max);
    return x;
}

void destroy_mat(Mat* mat)
{
    free(mat->mat);
    free(mat);
}

Mat* mat_t(Mat* A, int thread_count)
{
    Mat* A_T = create_mat(A->m, A->n);
    unsigned int i, j;


    for(i = 0; i < A->n; ++i)
        for(j = 0; j < A_T->n; ++j)
            A_T->mat[j*A_T->m+i] = A->mat[i*A->m+j];
    return A_T;
}

Mat* mat_mat_mult(Mat* A, Mat* B, int thread_count)
{
    if(A->m != B->n)
        return NULL;

    Mat* C = create_mat(A->n, B->m);

    unsigned i, j, k;

    for(i = 0; i < A->n; ++i)
        for(j = 0; j < B->m; ++j){
            double tmp = 0.0;
            for(k = 0; k < B->n; ++k)
                tmp += A->mat[i*A->m+k] * B->mat[k*B->m+j];
            C->mat[i*B->m+j] = tmp;
        }

    return C;
}

void mat_vect_mult(Mat* A, double* x, unsigned int x_size, double* y, int thread_count)
{
    if(A->m != x_size){
        printf("Error: dimensions of matrix\n");
        exit(0);
    }

    unsigned i, j;
    for(i = 0; i < A->n; ++i){
        double tmp = 0.0;
        for(j = 0; j < x_size; ++j)
            tmp += A->mat[i*x_size + j] * x[j];
        y[i] = tmp;
    }
}

Mat* mat_mat_sub(Mat* A, Mat* B, int thread_count)
{
    if(A->n != B->n || A->m != B->m)
        return NULL;
    Mat* C = create_mat(A->n, A->m);

    unsigned int i, size = A->n*A->m;

    for(i = 0; i < size; ++i)
        C->mat[i] = A->mat[i] - B->mat[i];

    return C;
}

void vect_vect_sub(double* x, double* y, unsigned int size, double *vect_c, int thread_count)
{
    unsigned int i;

    for(i = 0; i < size; ++i)
        vect_c[i] = x[i] - y[i];
}

double* scalar_vect_sub(double scalar, double* vect, unsigned int size, int thread_count)
{
    unsigned int i;
    double* vect_c = (double*)malloc(size * sizeof (double));

    for(i = 0; i < size; ++i)
        vect_c[i] = scalar - vect[i];

    return vect_c;
}

double* scalar_vect_multiply(double scalar, double* vect, unsigned int size, int thread_count)
{
    unsigned int i;
    double* vect_c = (double*)malloc(size * sizeof (double));

    for(i = 0; i < size; ++i)
        vect_c[i] = scalar * vect[i];

    return vect_c;
}

double* vect_mult_ew(double* x, double* y, unsigned int size, int thread_count) //elementwise
{
    double* vect_c = (double*) malloc(size * sizeof(double));
    unsigned int i;

    for(i = 0; i < size; ++i)
        vect_c[i] = x[i] * y[i];

    return vect_c;
}

void print_mat(Mat* mat)
{
    unsigned int i, j;
    for(i = 0; i < mat->n; ++i){
        for(j = 0; j < mat->m; ++j)
            printf("%lf ", mat->mat[i*mat->m+j]);
        printf("\n");
    }
}

Mat* mat_sigmoid(Mat* mat, int thread_count)
{
    Mat* mat_sig = create_mat(mat->n, mat->m);
    unsigned int i, len = mat->n * mat->m;

    for(i = 0; i < len; ++i)
        mat_sig->mat[i] = 1.0 / (1.0 + exp(- mat->mat[i]));
    return mat_sig;
}

Mat* mat_sigmoid_der(Mat* mat, int thread_count)      //derivative of sigmoid
{
    Mat* mat_der = create_mat(mat->n, mat->m);
    unsigned int i, size = mat->n * mat->m;
    double tmp;
    for(i = 0; i < size; ++i){
        tmp = 1.0 / (1.0 + exp(- mat->mat[i]));
        mat_der->mat[i] = tmp * (1.0 - tmp);
    }
    return mat_der;
}

double* vect_sigmoid_der(unsigned int size, double* vector, int thread_count)
{
    double* vect_der = (double*)malloc(size * sizeof (double));
    unsigned int i;
    double tmp;
    for(i = 0; i < size; ++i){
        tmp = 1.0 / (1.0 + exp(- vector[i]));
        vect_der[i] = tmp * (1.0 - tmp);
    }
    return vect_der;
}

double* vect_sigmoid(unsigned int size, double* vector, int thread_count)
{
    double* vec_sig = create_vector(size);
    unsigned int i;

    for(i = 0; i < size; ++i)
        vec_sig[i] = 1.0 / (1.0 + exp(- vector[i]));
    return vec_sig;
}

void print_vec(double* x, unsigned int n)
{
    unsigned int i;
    for(i = 0; i < n; ++i)
        printf("%lf\n", x[i]);
}

Mat* mat_mat_mult_serial(Mat* A, Mat* B)
{
    if(A->m != B->n)
        return NULL;

    Mat* C = create_mat(A->n, B->m);

    unsigned i, j, k;
    for(i = 0; i < A->n; ++i)
        for(j = 0; j < B->m; ++j){
            double tmp = 0.0;
            for(k = 0; k < B->n; ++k)
                tmp += A->mat[i*A->m+k] * B->mat[k*B->m+j];
            C->mat[i*B->m+j] = tmp;
        }
    return C;
}

double* mat_vect_mult_serial(Mat* A, double* x, unsigned int x_size)
{
    if(A->m != x_size)
        return NULL;

    double* C = (double*)malloc(A->n * sizeof (double));

    unsigned i, j;
    for(i = 0; i < A->n; ++i){
        double tmp = 0.0;
        for(j = 0; j < x_size; ++j)
            tmp += A->mat[i*x_size + j] * x[j];
        C[i] = tmp;
    }
    return C;
}
