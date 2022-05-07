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
#include "mpi.h"

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
	int i,j,k,m,cidx;
	int iter, niter, cp_idx;
	MPI_Status status;
	double error, tval, maxerr, tmpmaxerr, starttime, endtime, itertime;
	chunk *cp;
	int thisIndex, ierr, nblocks;

	// MPI prolog
	MPI_Init(&ac, &av);
	MPI_Comm_rank(MPI_COMM_WORLD, &thisIndex);
	MPI_Comm_size(MPI_COMM_WORLD, &nblocks);

	// checking number of parameters
	if (ac < 4) {
 		if (thisIndex == 0)
			printf("Usage: jacobi X Y Z [nIter].\n");
		MPI_Finalize();
		return 1;
	}

	// reading number of chunks per dimension
	NX = atoi(av[1]);
	NY = atoi(av[2]);
	NZ = atoi(av[3]);
	if (NX*NY*NZ != nblocks) {
		if (thisIndex == 0) 
			printf("%d x %d x %d != %d\n", NX,NY,NZ, nblocks);
		MPI_Finalize();
		return 2;
	}
	if (ac == 5)
		niter = atoi(av[4]);
	else
		niter = DEFAULT_ITERATIONS;

	// creating a chunk for this rank and computing its neighbors
	cp = new chunk;
	index3d(thisIndex, cp->xidx, cp->yidx, cp->zidx);
	cp->xp = index1d((cp->xidx+1)%NX,cp->yidx,cp->zidx);
	cp->xm = index1d((cp->xidx+NX-1)%NX,cp->yidx,cp->zidx);
	cp->yp = index1d(cp->xidx,(cp->yidx+1)%NY,cp->zidx);
	cp->ym = index1d(cp->xidx,(cp->yidx+NY-1)%NY,cp->zidx);
	cp->zp = index1d(cp->xidx,cp->yidx,(cp->zidx+1)%NZ);
	cp->zm = index1d(cp->xidx,cp->yidx,(cp->zidx+NZ-1)%NZ);

	// filling out chunk values
	for(i=1; i<=DIMZ; i++)
		for(j=1; j<=DIMY; j++)
			for(k=1; k<=DIMX; k++)
				cp->t[k][j][i] = DIMY*DIMX*(i-1) + DIMX*(j-2) + (k-1);

	// synchronizing all ranks
	MPI_Barrier(MPI_COMM_WORLD);
	starttime = MPI_Wtime();

	// setting error and running all iterations
	maxerr = 0.0;
	for(iter=1; iter<=niter; iter++) {
		maxerr = 0.0;

		// moving data on message buffers
		copyout(cp->sbxm, cp->t, 1, 1, 1, DIMY, 1, DIMZ);
		copyout(cp->sbxp, cp->t, DIMX, DIMX, 1, DIMY, 1, DIMZ);
		copyout(cp->sbym, cp->t, 1, DIMX, 1, 1, 1, DIMZ);
		copyout(cp->sbyp, cp->t, 1, DIMX, DIMY, DIMY, 1, DIMZ);
		copyout(cp->sbzm, cp->t, 1, DIMX, 1, DIMY, 1, 1);
		copyout(cp->sbzp, cp->t, 1, DIMX, 1, DIMY, DIMZ, DIMZ);

		MPI_Request rreq[6];
		MPI_Status rsts[6];

		// YOUR CODE GOES HERE
		// receiving outer slabs from neighbors, use MPI_Irecv with requests in variable rreq
		// for instance, cp->rbxp is the buffer of dimension DIMY*DIMZ to receive data from cp->xp
		MPI_Irecv(cp->rbxm, DIMY*DIMZ, MPI_DOUBLE, cp->xm, 0, MPI_COMM_WORLD, &rreq[0]);
		MPI_Irecv(cp->rbxp, DIMY*DIMZ, MPI_DOUBLE, cp->xp, 0, MPI_COMM_WORLD, &rreq[1]);
		MPI_Irecv(cp->rbym, DIMX*DIMZ, MPI_DOUBLE, cp->ym, 0, MPI_COMM_WORLD, &rreq[2]);
		MPI_Irecv(cp->rbyp, DIMX*DIMZ, MPI_DOUBLE, cp->yp, 0, MPI_COMM_WORLD, &rreq[3]);
		MPI_Irecv(cp->rbzm, DIMX*DIMY, MPI_DOUBLE, cp->zm, 0, MPI_COMM_WORLD, &rreq[4]);
		MPI_Irecv(cp->rbzp, DIMX*DIMY, MPI_DOUBLE, cp->zp, 0, MPI_COMM_WORLD, &rreq[5]);

		// YOUR CODE GOES HERE
		// sending all outer slabs of own chunk to neighbors
		// for instance, cp->sbxm is the buffer of dimension DIMY*DIMZ for cp->xm
		MPI_Isend(cp->sbxm, DIMY*DIMZ, MPI_DOUBLE, cp->xm, 0, MPI_COMM_WORLD, &rreq[0]);
		MPI_Isend(cp->sbxp, DIMY*DIMZ, MPI_DOUBLE, cp->xp, 0, MPI_COMM_WORLD, &rreq[1]);
		MPI_Isend(cp->sbym, DIMX*DIMZ, MPI_DOUBLE, cp->ym, 0, MPI_COMM_WORLD, &rreq[2]);
		MPI_Isend(cp->sbyp, DIMX*DIMZ, MPI_DOUBLE, cp->yp, 0, MPI_COMM_WORLD, &rreq[3]);
		MPI_Isend(cp->sbzm, DIMX*DIMY, MPI_DOUBLE, cp->zm, 0, MPI_COMM_WORLD, &rreq[4]);
		MPI_Isend(cp->sbzp, DIMX*DIMY, MPI_DOUBLE, cp->zp, 0, MPI_COMM_WORLD, &rreq[5]);

		// YOUR CODE GOES HERE
		// waiting for all receive operations, use a single MPI_Waitall with variables rreq and rsts
		MPI_Waitall(6, rreq, rsts);

		// moving data from messages to chunk
		copyin(cp->sbxm, cp->t, 0, 0, 1, DIMY, 1, DIMZ);
		copyin(cp->sbxp, cp->t, DIMX+1, DIMX+1, 1, DIMY, 1, DIMZ);
		copyin(cp->sbym, cp->t, 1, DIMX, 0, 0, 1, DIMZ);
		copyin(cp->sbyp, cp->t, 1, DIMX, DIMY+1, DIMY+1, 1, DIMZ);
		copyin(cp->sbzm, cp->t, 1, DIMX, 1, DIMY, 0, 0);
		copyin(cp->sbzp, cp->t, 1, DIMX, 1, DIMY, DIMZ+1, DIMZ+1);
	
		// relaxation code
		for(i=1; i<=DIMZ; i++)
			for(j=1; j<=DIMY; j++)
				for(k=1; k<=DIMX; k++) {
					tval = (cp->t[k][j][i] + cp->t[k][j][i+1] +
						cp->t[k][j][i-1] + cp->t[k][j+1][i]+ 
						cp->t[k][j-1][i] + cp->t[k+1][j][i] + cp->t[k-1][j][i])/7.0;
					error = abs(tval-cp->t[k][j][i]);
					cp->t[k][j][i] = tval;
					if (error > maxerr) maxerr = error;
				}

		// YOUR CODE GOES HERE	
		// all reduce to find maximum error, contribution goes out in variable maxerr, while result gets into tmpmaxerr
        // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
		MPI_Allreduce(&maxerr, &tmpmaxerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		maxerr = tmpmaxerr;

		// timing execution
		endtime = MPI_Wtime();
		itertime = endtime - starttime;
		double  it;

		// YOUR CODE GOES HERE
		// all reduce to sum iteration time, contribution goes out in variable itertime, while result gets into it
        MPI_Allreduce(&itertime, &it, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// printing average iteration time
		itertime = it/nblocks;
		if (thisIndex == 0)
			printf("Iteration %d elapsed time: %f time: %lf maxerr: %lf\n", iter, MPI_Wtime(), itertime, maxerr);
		starttime = MPI_Wtime();
	}

	// MPI epilog
	MPI_Finalize();
	return 0;
}
