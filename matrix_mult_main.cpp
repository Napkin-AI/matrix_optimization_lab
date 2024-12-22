#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <cstring>
#include <immintrin.h>

#define MKL
#ifdef MKL
#include "mkl.h"
#endif

using namespace std;

void generation(double * mat, size_t size)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> uniform_distance(-2.001, 2.001);
	for (size_t i = 0; i < size * size; i++)
		mat[i] = uniform_distance(gen);
}

// void matrix_mult1(double * a, double * b, double * res, size_t size)
// {
// #pragma omp parallel for
// 	for (int i = 0; i < size; i++)
// 	{
// 		for (int j = 0; j < size; j++)
// 	    {
// 			for (int k = 0; k < size; k++)
// 		    {
// 				res[i*size + j] += a[i*size + k] * b[k*size + j];
// 			}
// 		}
// 	}
// }
// void matrix_mult(const double* a, const double* b, double* res, size_t size) {
//     const size_t block_size = 20;
//     #pragma omp parallel for
//     for (size_t i = 0; i < size; i += block_size) {
//         for (size_t j = 0; j < size; j += block_size) {
//             for (size_t k = 0; k < size; k += block_size) {
//                 for (size_t i1 = i; i1 < i + block_size; i1++) {
//                     for (size_t k1 = k; k1 < k + block_size; k1++) {
//                         double tmp = a[i1 * size + k1];
//                         for (size_t j1 = j; j1 < j + block_size; j1 += 4) {
//                             __m256d rb = _mm256_loadu_pd(&b[k1 * size + j1]);
//                             __m256d rres = _mm256_loadu_pd(&res[i1 * size + j1]);
//                             rres = _mm256_fmadd_pd(_mm256_set1_pd(tmp), rb, rres);
//                             _mm256_storeu_pd(&res[i1 * size + j1], rres);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// void matrix_mult(const double* a, const double* b, double* res, size_t size) {
//     const size_t block_size = 20;
//     #pragma omp parallel for
//     for (size_t i = 0; i < size; i += block_size) {
//         for (size_t j = 0; j < size; j += block_size) {
//             for (size_t k = 0; k < size; k += block_size) {
//                 for (size_t i1 = i; i1 < i + block_size; i1++) {
//                     for (size_t k1 = k; k1 < k + block_size; k1++) {
// 						__m256d ra = _mm256_set1_pd(a[i1 * size + k1]);

//                         for (size_t j1 = j; j1 < j + block_size; j1 += 4) {

//                             __m256d rb = _mm256_loadu_pd(&b[k1 * size + j1]);
//                             __m256d rres = _mm256_loadu_pd(&res[i1 * size + j1]);
//                             rres = _mm256_fmadd_pd(ra, rb, rres);
//                             _mm256_storeu_pd(&res[i1 * size + j1], rres);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

void matrix_mult(const double* a, const double* b, double* res, size_t size) {
    const size_t block_size = 20;
    #pragma omp parallel for
    for (size_t i = 0; i < size; i += block_size) {
        for (size_t j = 0; j < size; j += block_size) {
            for (size_t k = 0; k < size; k += block_size) {
                for (size_t i1 = i; i1 < i + block_size; i1++) {
                    for (size_t k1 = k; k1 < k + block_size; k1++) {

						__m256d ra = _mm256_set1_pd(a[i1 * size + k1]);
						__m256d rb = _mm256_loadu_pd(&b[k1 * size + j]);
                        __m256d rres = _mm256_loadu_pd(&res[i1 * size + j]);
                        _mm256_storeu_pd(&res[i1 * size + j], _mm256_fmadd_pd(ra, rb, rres));

						rb = _mm256_loadu_pd(&b[k1 * size + j + 4]);
                     	rres = _mm256_loadu_pd(&res[i1 * size + j + 4]);
                        _mm256_storeu_pd(&res[i1 * size + j + 4], _mm256_fmadd_pd(ra, rb, rres));

						rb = _mm256_loadu_pd(&b[k1 * size + j + 8]);
                     	rres = _mm256_loadu_pd(&res[i1 * size + j + 8]);
                        _mm256_storeu_pd(&res[i1 * size + j + 8], _mm256_fmadd_pd(ra, rb, rres));

						rb = _mm256_loadu_pd(&b[k1 * size + j + 12]);
                     	rres = _mm256_loadu_pd(&res[i1 * size + j + 12]);
                        _mm256_storeu_pd(&res[i1 * size + j + 12], _mm256_fmadd_pd(ra, rb, rres));

						rb = _mm256_loadu_pd(&b[k1 * size + j + 16]);
                     	rres = _mm256_loadu_pd(&res[i1 * size + j + 16]);
                        _mm256_storeu_pd(&res[i1 * size + j + 16], _mm256_fmadd_pd(ra, rb, rres));

                    }
                }
            }
        }
    }
}

int main()
{
	double *mat, *mat_mkl, *a, *b, *a_mkl, *b_mkl;
	size_t size = 1000;
	chrono::time_point<chrono::system_clock> start, end;

	mat = new double[size * size];
	a = new double[size * size];
	b = new double[size * size];
	generation(a, size);
	generation(b, size);
	memset(mat, 0, size*size * sizeof(double));

#ifdef MKL     
    mat_mkl = new double[size * size];
	a_mkl = new double[size * size];
	b_mkl = new double[size * size];
	memcpy(a_mkl, a, sizeof(double)*size*size);
    memcpy(b_mkl, b, sizeof(double)*size*size);
	memset(mat_mkl, 0, size*size * sizeof(double));
#endif

	start = chrono::system_clock::now();
	matrix_mult(a, b, mat, size);
	end = chrono::system_clock::now();
    
   
	int elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time: " << elapsed_seconds/1000.0 << " sec" << endl;

#ifdef MKL 
	start = chrono::system_clock::now();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0, a_mkl, size, b_mkl, size, 0.0, mat_mkl, size);
    end = chrono::system_clock::now();
    
    elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time mkl: " << elapsed_seconds/1000.0 << " sec" << endl;
     
    int flag = 0;
    for (unsigned int i = 0; i < size * size; i++)
        if(abs(mat[i] - mat_mkl[i]) > size*1e-14){
		    flag = 1;
        }
    if (flag){
		for(int i = 0; i < 10; ++i)
		{
			for(int j = 0; j < 10; ++j)
			{
				std::cout << mat[i*size+j] << " ";
			}
			cout << '\n';
		}
		for(int i = 0; i < 10; ++i)
		{
			for(int j = 0; j < 10; ++j)
			{
				std::cout << mat_mkl[i*size+j] << " ";
			}
			cout << '\n';
		}	
        cout << "fail" << endl;
	}
    else
        cout << "correct" << endl; 
    
    delete (a_mkl);
    delete (b_mkl);
    delete (mat_mkl);
#endif

    delete (a);
    delete (b);
    delete (mat);

	//system("pause");
	
	return 0;
}
