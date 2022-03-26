/**
 * Costa Rica Institute of Technology
 * School of Computing
 * Parallel Computing (MC-8836)
 * Instructor Esteban Meneses, PhD (esteban.meneses@acm.org)
 * OpenMP parallel shear sort.
 */

#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include "timer.h"
#include "io.h"

#define MAX_VALUE 10000

#include <bits/stdc++.h>

using namespace std;

void rowsort(int i, int **a, int n){
    for(int j=0;j<n-1;j++){
        for(int k=0;k<n-j-1;k++){
            if(a[i][k]>a[i][k+1]){
                int temp=a[i][k];
                a[i][k]=a[i][k+1];
                a[i][k+1]=temp;
            }
        }
    }
}
void rowrevsort(int i, int **a, int n){
    for(int j=0;j<n-1;j++){
        for(int k=0;k<n-j-1;k++){
            if(a[i][k]<a[i][k+1]){
                int temp=a[i][k];
                a[i][k]=a[i][k+1];
                a[i][k+1]=temp;
            }
        }
    }
}
void colsort(int i, int **a, int n){
    for(int j=0;j<n-1;j++){
        for(int k=0;k<n-j-1;k++){
            if(a[k][i]>a[k+1][i]){
                int temp=a[k][i];
                a[k][i]=a[k+1][i];
                a[k+1][i]=temp;
            }
        }
    }
}

// Shear sort function
void shear_sort(int **a, int n){
	
	int m=(int)ceil(log2(n));
    for(int i=0;i<m;i++){
        #pragma omp parallel for shared(a)
        for(int j=0;j<n;j++){
            if(j%2==0){
                rowsort(j,a,n);
            }else{
                rowrevsort(j,a,n);
            }
        }
        #pragma omp parallel for shared(a)
        for(int j=0;j<n;j++) colsort(j,a,n);
            cout<<endl;
    }
        
    #pragma omp parallel for shared(a)
    for(int j=0;j<n;j++){
        if(j%2==0){
        rowsort(j,a,n);
        }else{
            rowrevsort(j,a,n);
        }
    }
}

// Main method      
int main(int argc, char* argv[]) {
	int N, M;
	int **A;
	double elapsedTime;

	// checking parameters
	if (argc != 2 && argc != 3) {
		cout << "Parameters: <N> [<file>]" << endl;
		return 1;
	}
	N = atoi(argv[1]);
	M = (int) sqrt(N); 
	if(N != M*M){
		cout << "N has to be a perfect square!" << endl;
		exit(1);
	}	

	// allocating matrix A
	A = new int*[M];
	for (int i=0; i<M; i++){
		A[i] = new int[M];
	}

	// reading files (optional)
	if(argc == 3){
		readMatrixFile(A,M,argv[2]);
	} else {
		srand48(time(NULL));
		#pragma omp parallel for
		for(int i=0; i<M; i++){
			#pragma omp parallel for
			for(int j=0; j<M; j++){
				A[i][j] = lrand48() % MAX_VALUE;
			}
		}
	}
	
	// starting timer
	timerStart();

	// calling shear sort function
	shear_sort(A,M);

	// testing the results is correct
	if(argc == 3){
		printMatrix(A,M);
	}
	
	// stopping timer
	elapsedTime = timerStop();

	cout << "Duration: " << elapsedTime << " seconds" << std::endl;

	// releasing memory
	#pragma omp parallel for shared(A)
	for (int i=0; i<M; i++) {
		delete [] A[i];
	}
	delete [] A;

	return 0;	
}
