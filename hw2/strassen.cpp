/**
 * Costa Rica Institute of Technology
 * School of Computing
 * MC-8836: Parallel Computing
 * Instructor Esteban Meneses, PhD (esteban.meneses@acm.org)
 * Student: 
 * Cilk Plus parallel Strassen algorithm for matrix multiplication.
 */

#include <cstdio>
#include <cstdlib>
#include <cilk/cilk.h>
#ifdef CILK_SERIALIZE
#include <cilk/cilk_stub.h>
#endif
#include "timer.h"
#include "io.h"
#include <algorithm>
#include <iostream>
#include <bits/stdc++.h>
#include <vector>
using namespace std;

int nextpowerof2(int k){
    return pow(2, int(ceil(log2(k))));
}

void display(vector< vector<int>> &matrix, int m, int n){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            if (j != 0){
                cout << "\t";
            }
            cout << matrix[i][j];
        }
        cout << endl;
    }
}

void add(vector<vector<int>> &A, vector<vector<int>> &B, vector<vector<int>> &C, int size){
    cilk_for (int i = 0; i < size; i++){
        cilk_for (int j = 0; j < size; j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void sub(vector<vector<int>> &A, vector<vector<int>> &B, vector<vector<int>> &C, int size){
    cilk_for (int i = 0; i < size; i++){
        cilk_for (int j = 0; j < size; j++){
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void Strassen_algorithmA(vector<vector<int>> &A, vector<vector<int>> &B, vector<vector<int>> &C, int size)
{
    //base case
    if (size == 1)
    {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }
    else
    {
        int new_size = size / 2;
        vector<int> z(new_size);
        vector<vector<int>>
            a11(new_size, z), a12(new_size, z), a21(new_size, z), a22(new_size, z),
            b11(new_size, z), b12(new_size, z), b21(new_size, z), b22(new_size, z),
            c11(new_size, z), c12(new_size, z), c21(new_size, z), c22(new_size, z),
            p1(new_size, z), p2(new_size, z), p3(new_size, z), p4(new_size, z),
            p5(new_size, z), p6(new_size, z), p7(new_size, z),
            a1Result(new_size, z), b1Result(new_size, z),
            a2Result(new_size, z),
            b3Result(new_size, z),
            b4Result(new_size, z),
            a5Result(new_size, z),
            a6Result(new_size, z), b6Result(new_size, z),
            a7Result(new_size, z), b7Result(new_size, z);
        
        //dividing the matrices into sub-matrices:
        cilk_for (int i = 0; i < new_size; i++)
            {
                cilk_for (int j = 0; j < new_size; j++)
                {
                    a11[i][j] = A[i][j];
                    a12[i][j] = A[i][j + new_size];
                    a21[i][j] = A[i + new_size][j];
                    a22[i][j] = A[i + new_size][j + new_size];
        
                    b11[i][j] = B[i][j];
                    b12[i][j] = B[i][j + new_size];
                    b21[i][j] = B[i + new_size][j];
                    b22[i][j] = B[i + new_size][j + new_size];
                }
            }

        // Calculating p1 to p7:

        cilk_spawn add(a11, a22, a1Result, new_size);     // a11 + a22
        add(b11, b22, b1Result, new_size);    // b11 + b22
        cilk_sync;
        cilk_spawn Strassen_algorithmA(a1Result, b1Result, p1, new_size); 
        // p1 = (a11+a22) * (b11+b22)

        add(a21, a22, a2Result, new_size); // a21 + a22
        cilk_spawn Strassen_algorithmA(a2Result, b11, p2, new_size);
        // p2 = (a21+a22) * (b11)

        sub(b12, b22, b3Result, new_size);      // b12 - b22
        cilk_spawn Strassen_algorithmA(a11, b3Result, p3, new_size);
        // p3 = (a11) * (b12 - b22)

        sub(b21, b11, b4Result, new_size);       // b21 - b11
        cilk_spawn Strassen_algorithmA(a22, b4Result, p4, new_size); 
        // p4 = (a22) * (b21 - b11)

        add(a11, a12, a5Result, new_size);      // a11 + a12
        cilk_spawn Strassen_algorithmA(a5Result, b22, p5, new_size);
        // p5 = (a11+a12) * (b22)

        sub(a21, a11, a6Result, new_size);      // a21 - a11
        add(b11, b12, b6Result, new_size);               
        // b11 + b12https://www.onlinegdb.com/online_c_compiler#tab-stdin
        cilk_spawn Strassen_algorithmA(a6Result, b6Result, p6, new_size);
        // p6 = (a21-a11) * (b11+b12)

        sub(a12, a22, a7Result, new_size);      // a12 - a22
        add(b21, b22, b7Result, new_size);                
        // b21 + b22
        cilk_spawn Strassen_algorithmA(a7Result, b7Result, p7, new_size);
        // p7 = (a12-a22) * (b21+b22)
        
        cilk_sync;

        // calculating c21, c21, c11 e c22:

        cilk_spawn add(p3, p5, c12, new_size); // c12 = p3 + p5
        cilk_spawn add(p2, p4, c21, new_size); // c21 = p2 + p4

        
        add(p1, p4, a1Result, new_size);       // p1 + p4
        add(a1Result, p7, b1Result, new_size);  // p1 + p4 + p7
        cilk_spawn sub(b1Result, p5, c11, new_size); // c11 = p1 + p4 - p5 + p7
        

        add(p1, p3, a6Result, new_size);       // p1 + p3
        add(a6Result, p6, b6Result, new_size);  // p1 + p3 + p6
        sub(b6Result, p2, c22, new_size); // c22 = p1 + p3 - p2 + p6
        
        cilk_sync;

        // Grouping the results obtained in a single matrix:
        cilk_for (int i = 0; i < new_size; i++)
        {
            cilk_for (int j = 0; j < new_size; j++)
            {
                C[i][j] = c11[i][j];
                C[i][j + new_size] = c12[i][j];
                C[i + new_size][j] = c21[i][j];
                C[i + new_size][j + new_size] = c22[i][j];
            }
        }
    }
}

void Strassen_algorithm(vector<vector<int>> &A, vector<vector<int>> &B, vector<vector<int>> &C, int N)
{  
/* Check to see if these matrices are already square and have dimensions of a power of 2. If not,
 * the matrices must be resized and padded with zeroes to meet this criteria. */
    int k = N;

    int s = nextpowerof2(k);

    vector<int> z(s);
    vector<vector<int>> Aa(s, z), Bb(s, z), Cc(s, z);

    cilk_for (unsigned int i = 0; i < k; i++)
    {
        cilk_for (unsigned int j = 0; j < k; j++)
        {
            Aa[i][j] = A[i][j];
        }
    }
    cilk_for (unsigned int i = 0; i < k; i++)
    {
        cilk_for (unsigned int j = 0; j < k; j++)
        {
            Bb[i][j] = B[i][j];
        }
    }
    Strassen_algorithmA(Aa, Bb, Cc, s);
    
    cilk_for (unsigned int i = 0; i < k; i++)
    {
        cilk_for (unsigned int j = 0; j < k; j++)
        {
            C[i][j] = Cc[i][j];
        }
    }
    //display(C, m, b);
}

// Main method      
int main(int argc, char* argv[]) {
	int N;
	int **A, **B, **C;
	double elapsedTime;

	// checking parameters
	if (argc != 2 && argc != 4) {
		cout << "Parameters: <N> [<fileA> <fileB>]" << endl;
		return 1;
	}
	N = atoi(argv[1]);

	// allocating matrices
	A = new int*[N];
	B = new int*[N];
	C = new int*[N];
	for (int i=0; i<N; i++){
		A[i] = new int[N];
		B[i] = new int[N];
		C[i] = new int[N];
	}

	// reading files (optional)
	if(argc == 4){
		readMatrixFile(A,N,argv[2]);
		readMatrixFile(B,N,argv[3]);
	}

	// starting timer
	timerStart();

	// YOUR CODE GOES HERE
	
	vector<int> z(N);
    vector<vector<int>>
        vA(N, z), vB(N, z), vC(N, z);
        
    for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                vA[i][j] = A[i][j];
                vB[i][j] = B[i][j];
                vC[i][j] = C[i][j];
            }
        }
        
    //display(vA,N,N);
    //cout << vA.size();
    //cout << endl;
    //cout << vA[0].size();
    //cout << endl;
	
	Strassen_algorithm(vA, vB, vC, N);
	
	for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i][j] = vC[i][j];
            }
        }

	// testing the results is correct
	if(argc == 4){
		printMatrix(C,N);
	}
	
	// stopping timer
	elapsedTime = timerStop();

	cout << "Duration: " << elapsedTime << " seconds" << std::endl;

	// releasing memory
	for (int i=0; i<N; i++) {
		delete [] A[i];
		delete [] B[i];
		delete [] C[i];
	}
	delete [] A;
	delete [] B;
	delete [] C;

	return 0;	
}

