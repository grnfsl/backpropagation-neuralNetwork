typedef struct Mat{
    unsigned int n;
    unsigned int m;
    double* mat;
} Mat;

Mat* create_mat(unsigned int n, unsigned int m);
Mat* create_random_mat(unsigned int n, unsigned int m, double min, double max);
void destroy_mat(Mat* mat);

double* create_vector(unsigned int n);
double* create_random_vector(unsigned int n, double min, double max);

Mat* mat_mat_mult(Mat* A, Mat* B, int thread_count);
void mat_vect_mult(Mat* A, double* x, unsigned int x_size, double* y, int thread_count);
Mat* mat_mat_sub(Mat* A, Mat* B, int thread_count);

double* scalar_vect_sub(double scalar, double* vect, unsigned int size, int thread_count);
double* scalar_vect_multiply(double scalar, double* vect, unsigned int size, int thread_count);
double* vect_mult_ew(double* x, double* y, unsigned int size, int thread_count); //elementwise
void vect_vect_sub(double* x, double* y, unsigned int size, double* vect_c, int thread_count);

void print_mat(Mat* mat);
void print_vec(double* x, unsigned int n);

Mat* mat_t(Mat* A, int thread_count);

Mat* mat_mat_mult_serial(Mat* A, Mat* B);
double* mat_vect_mult_serial(Mat* A, double* x, unsigned int x_size);

Mat* mat_sigmoid(Mat* mat, int thread_count);
Mat* mat_sigmoid_der(Mat* mat, int thread_count);
double* vect_sigmoid(unsigned int size, double* vector, int thread_count);
double* vect_sigmoid_der(unsigned int size, double* vector, int thread_count);
