/**
 * Costa Rica Institute of Technology
 * School of Computing
 * Parallel Computing (MC-8836)
 * Instructor Esteban Meneses, PhD (esteban.meneses@acm.org)
 * MPI Jacobi relaxation code for 3D (original version from
 * the Parallel Programming Laboratory of the University of 
 * Illinois at Urbana-Champaign)
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define DIMX 128
#define DIMY 128
#define DIMZ 128

#define DEFAULT_ITERATIONS 20

int NX, NY, NZ;

class chunk {
public:
	double t[DIMX+2][DIMY+2][DIMZ+2];
	int xidx, yidx, zidx;
	int xm, xp, ym, yp, zm, zp;
	double sbxm[DIMY*DIMZ];
	double sbxp[DIMY*DIMZ];
	double sbym[DIMX*DIMZ];
	double sbyp[DIMX*DIMZ];
	double sbzm[DIMX*DIMY];
	double sbzp[DIMX*DIMY];
	double rbxm[DIMY*DIMZ];
	double rbxp[DIMY*DIMZ];
	double rbym[DIMX*DIMZ];
	double rbyp[DIMX*DIMZ];
	double rbzm[DIMX*DIMY];
	double rbzp[DIMX*DIMY];
};


#define abs(x) ((x)<0.0 ? -(x) : (x))

// Function to translate (x,y,z) coordinates into a single index position
int index1d(int ix, int iy, int iz)
{
	return NY*NZ*ix + NZ*iy + iz;
}

// Function to translate a single index position into (x,y,z) coordinates
void index3d(int index, int& ix, int& iy, int& iz)
{
	ix = index/(NY*NZ);
	iy = (index%(NY*NZ))/NZ;
	iz = index%NZ;
}

// Function to copy data from chunk to message buffer
static void copyout(double *d, double t[DIMX+2][DIMY+2][DIMZ+2],
                    int sx, int ex, int sy, int ey, int sz, int ez)
{
	int i, j, k;
	int l = 0;
	for(i=sx; i<=ex; i++)
		for(j=sy; j<=ey; j++)
			for(k=sz; k<=ez; k++, l++)
				d[l] = t[i][j][k];
}

static void copyArray(int dimA, int dimB, double x[], double y[]){
	int size = dimA * dimB;
	for (int i = 0; i<size; i++){
		x[i] = y[i];
	}
}

// Function to copy data from message buffer into chunk
static void copyin(double *d, double t[DIMX+2][DIMY+2][DIMZ+2],
                    int sx, int ex, int sy, int ey, int sz, int ez)
{
	int i, j, k;
	int l = 0;
	for(i=sx; i<=ex; i++)
		for(j=sy; j<=ey; j++)
			for(k=sz; k<=ez; k++, l++)
				t[i][j][k] = d[l];
}

int main(int ac, char** av)
{
	int i,j,k,cidx;
	int iter, niter, cp_idx;
	double error, tval, maxerr, tmpmaxerr, starttime, endtime, itertime;
	int thisIndex, ierr, nblocks;


	// checking number of parameters
	if (ac < 4) {
		printf("Usage: jacobi X Y Z [nIter].\n");
		return 1;
	}

	// reading number of chunks per dimension
	NX = atoi(av[1]);
	NY = atoi(av[2]);
	NZ = atoi(av[3]);
	nblocks = NX*NY*NZ;
	
	if (ac == 5)
		niter = atoi(av[4]);
	else
		niter = DEFAULT_ITERATIONS;

	// creating a chunk for this rank and computing its neighbors
	chunk * cp[NX][NY][NY];
	
	for(i=1; i<=NZ; i++)
		for(j=1; j<=NY; j++)
			for(k=1; k<=NX; k++)
			    cp[k][j][i] = new chunk;
	
	for(i=1; i<=NZ; i++)
		for(j=1; j<=NY; j++)
			for(k=1; k<=NX; k++)
			    index3d(i+j+k, cp[k][j][i]->xidx, cp[k][j][i]->yidx, cp[k][j][i]->zidx);
	            cp[k][j][i]->xp = index1d((cp[k][j][i]->xidx+1)%NX,cp[k][j][i]->yidx,cp[k][j][i]->zidx);
	            cp[k][j][i]->xm = index1d((cp[k][j][i]->xidx+NX-1)%NX,cp[k][j][i]->yidx,cp[k][j][i]->zidx);
	            cp[k][j][i]->yp = index1d(cp[k][j][i]->xidx,(cp[k][j][i]->yidx+1)%NY,cp[k][j][i]->zidx);
	            cp[k][j][i]->ym = index1d(cp[k][j][i]->xidx,(cp[k][j][i]->yidx+NY-1)%NY,cp[k][j][i]->zidx);
	            cp[k][j][i]->zp = index1d(cp[k][j][i]->xidx,cp[k][j][i]->yidx,(cp[k][j][i]->zidx+1)%NZ);
	            cp[k][j][i]->zm = index1d(cp[k][j][i]->xidx,cp[k][j][i]->yidx,(cp[k][j][i]->zidx+NZ-1)%NZ);

	            // filling out chunk values
	            for(int m=1; m<=DIMZ; m++)
	            	for(int n=1; n<=DIMY; n++)
	            		for(int p=1; p<=DIMX; p++)
	            			cp[k][j][i]->t[p][n][m] = DIMY*DIMX*(m-1) + DIMX*(n-2) + (p-1);


	// setting error and running all iterations
	maxerr = 0.0;
	for(iter=1; iter<=niter; iter++) {
		maxerr = 0.0;
		double maxerr_local;
		double itersum;
		
		for(i=1; i<=NZ; i++)
		    for(j=1; j<=NY; j++)
			    for(k=1; k<=NX; k++) {
		
		            std::chrono::steady_clock::time_point starttime = std::chrono::steady_clock::now();
		            
		            maxerr_local = 0;
		            
		            // moving data on message buffers
		            copyout(cp[k][j][i]->sbxm, cp[k][j][i]->t, 1, 1, 1, DIMY, 1, DIMZ);
		            copyout(cp[k][j][i]->sbxp, cp[k][j][i]->t, DIMX, DIMX, 1, DIMY, 1, DIMZ);
		            copyout(cp[k][j][i]->sbym, cp[k][j][i]->t, 1, DIMX, 1, 1, 1, DIMZ);
		            copyout(cp[k][j][i]->sbyp, cp[k][j][i]->t, 1, DIMX, DIMY, DIMY, 1, DIMZ);
		            copyout(cp[k][j][i]->sbzm, cp[k][j][i]->t, 1, DIMX, 1, DIMY, 1, 1);
		            copyout(cp[k][j][i]->sbzp, cp[k][j][i]->t, 1, DIMX, 1, DIMY, DIMZ, DIMZ);
		            
		            int a, b, c;
		            
		            index3d(cp[k][j][i]->xm, a, b, c);
		            //copyArray(DIMY, DIMZ, cp[k][j][i]->rbxm, cp[a][b][c]->sbxm);
		            
		            index3d(cp[k][j][i]->xp, a, b, c);
		            //copyArray(DIMY, DIMZ, cp[k][j][i]->rbxp, cp[a][b][c]->sbxp);
		            
		            index3d(cp[k][j][i]->ym, a, b, c);
		            //copyArray(DIMX, DIMZ, cp[k][j][i]->rbym, cp[a][b][c]->sbym);
		            
		            index3d(cp[k][j][i]->yp, a, b, c);
		            //copyArray(DIMX, DIMZ, cp[k][j][i]->rbyp, cp[a][b][c]->sbyp);
		            
		            index3d(cp[k][j][i]->zm, a, b, c);
		            //copyArray(DIMY, DIMX, cp[k][j][i]->rbzm, cp[a][b][c]->sbzm);
		            
		            index3d(cp[k][j][i]->zp, a, b, c);
		            //copyArray(DIMY, DIMX, cp[k][j][i]->rbzp, cp[a][b][c]->sbzp);
		            
		            
		            
		            // moving data from messages to chunk
		            copyin(cp[k][j][i]->sbxm, cp[k][j][i]->t, 0, 0, 1, DIMY, 1, DIMZ);
		            copyin(cp[k][j][i]->sbxp, cp[k][j][i]->t, DIMX+1, DIMX+1, 1, DIMY, 1, DIMZ);
		            copyin(cp[k][j][i]->sbym, cp[k][j][i]->t, 1, DIMX, 0, 0, 1, DIMZ);
		            copyin(cp[k][j][i]->sbyp, cp[k][j][i]->t, 1, DIMX, DIMY+1, DIMY+1, 1, DIMZ);
		            copyin(cp[k][j][i]->sbzm, cp[k][j][i]->t, 1, DIMX, 1, DIMY, 0, 0);
		            copyin(cp[k][j][i]->sbzp, cp[k][j][i]->t, 1, DIMX, 1, DIMY, DIMZ+1, DIMZ+1);
		            
		            // relaxation code
		            for(i=1; i<=DIMZ; i++)
		            	for(j=1; j<=DIMY; j++)
		            		for(k=1; k<=DIMX; k++) {
		            			tval = (cp[k][j][i]->t[k][j][i] + cp[k][j][i]->t[k][j][i+1] +
		            				cp[k][j][i]->t[k][j][i-1] + cp[k][j][i]->t[k][j+1][i]+ 
		            				cp[k][j][i]->t[k][j-1][i] + cp[k][j][i]->t[k+1][j][i] + cp[k][j][i]->t[k-1][j][i])/7.0;
		            			error = abs(tval-cp[k][j][i]->t[k][j][i]);
		            			cp[k][j][i]->t[k][j][i] = tval;
		            			if (error > maxerr_local) maxerr_local = error;
		            		}
		            
		            if (maxerr_local > maxerr){
		                maxerr = maxerr_local;
		            }
		            
		            // timing execution
		            std::chrono::steady_clock::time_point endtime = std::chrono::steady_clock::now();
		            itertime = std::chrono::duration_cast<std::chrono::seconds>(endtime - starttime).count();
		            itersum += itertime;
		            
                }

		// printing average iteration time
		itertime = itersum/nblocks;
		printf("Iteration %d time: %lf maxerr: %lf\n", iter, itertime, maxerr);
	}

	return 0;
}

