#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>


// Check if the given command has returned an error.
#define CUDA_CHECK(cmd) if ((cmd) != cudaSuccess) { \
    printf("ERROR: cuda error at %s:%d\n", __FILE__, __LINE__); abort(); }

#define DEBUG std::cout << "** DEBUG " << __LINE__ << " **" << std::endl


// Constants.
#define BLOCK_SIZE 32
#define PARRSZ (sizeof(FPpart) * 7)


// Pointer allocated to be used by the CPU.
struct grid *gGpuGrid;
struct parameters *gGpuParam;
struct EMfield *gGpuField;
struct particles *gGpuPart;


// Allocate and copy data to the GPU.
static void CudaAllocateAndCopy(void *dest, const void *src, long long size)
{
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    CUDA_CHECK(cudaMemcpy(dest, &ptr, sizeof(void *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptr, src, size, cudaMemcpyHostToDevice));
}



/// Allocate particle arrays
/// ------------------------
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    // *Note*: all following parameters are constants.

    part->species_ID = is;  // set species ID.
    part->nop = param->np[is];  // Number of particles.
    part->npmax = param->npMax[is];  // Maximum number of particles.

    // Choose a different number of mover iterations for ions and electrons.
    if (param->qom[is] < 0){  // Electrons.
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration.
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }

    // Particles per cell.
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // Cast it to required precision.
    part->qom = (FPpart) param->qom[is];
    
    // Initialize drift and thermal velocities drift.
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];


    // /!\ W A R N I N G
    // The following code assumes that FPpart == FPinterp

    // Use only one array so we only need one copy.
    auto npmax = part->npmax;
    FPpart *array = new FPpart[npmax * 7];

    part->x = array + (npmax * 0);
    part->y = array + (npmax * 1);
    part->z = array + (npmax * 2);

    // Allocate velocity.
    part->u = array + (npmax * 3);
    part->v = array + (npmax * 4);
    part->w = array + (npmax * 5);

    // Allocate charge = q * statistical weight.
    part->q = array + (npmax * 6);
}



/// Deallocate particle arrays
/// --------------------------
void particle_deallocate(struct particles* part)
{
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}



/// Initialize GPU data
/// -------------------
void particle_init_gpu(particles *part, grid *grd, parameters *param, EMfield *field)
{
    // Allocate and copy parameters to the GPU.
    CUDA_CHECK(cudaMalloc(&gGpuParam, sizeof(parameters)));
    CUDA_CHECK(cudaMemcpy(gGpuParam, param, sizeof(parameters), cudaMemcpyHostToDevice));

    int size = grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield);


    // Allocate and copy the grid to the GPU.
    CUDA_CHECK(cudaMalloc(&gGpuGrid, sizeof(grid)));
    CUDA_CHECK(cudaMemcpy(gGpuGrid, grd, sizeof(grid), cudaMemcpyHostToDevice));

    CudaAllocateAndCopy(&gGpuGrid->XN_flat, grd->XN_flat, size);
    CudaAllocateAndCopy(&gGpuGrid->YN_flat, grd->YN_flat, size);
    CudaAllocateAndCopy(&gGpuGrid->ZN_flat, grd->ZN_flat, size);


    // Copy the field to the GPU.
    CUDA_CHECK(cudaMalloc(&gGpuField, sizeof(EMfield)));
    CUDA_CHECK(cudaMemcpy(gGpuField, field, sizeof(EMfield), cudaMemcpyHostToDevice));

    CudaAllocateAndCopy(&gGpuField->Ex_flat, field->Ex_flat, size);
    CudaAllocateAndCopy(&gGpuField->Ey_flat, field->Ey_flat, size);
    CudaAllocateAndCopy(&gGpuField->Ez_flat, field->Ez_flat, size);
    CudaAllocateAndCopy(&gGpuField->Bxn_flat, field->Bxn_flat, size);
    CudaAllocateAndCopy(&gGpuField->Byn_flat, field->Byn_flat, size);
    CudaAllocateAndCopy(&gGpuField->Bzn_flat, field->Bzn_flat, size);


    // Allocate and copy particules array.
    CUDA_CHECK(cudaMalloc(&gGpuPart, sizeof(particles) * param->ns));
    CUDA_CHECK(cudaMemcpy(gGpuPart, part, sizeof(particles) * param->ns, cudaMemcpyHostToDevice));

    for (int is = 0; is < param->ns; is++) {
        CUDA_CHECK(cudaMalloc(&part[is].GPU_array, PARRSZ * part[is].npmax));
        CUDA_CHECK(cudaMemcpy(&gGpuPart[is].x, &part[is].GPU_array, sizeof(void *), cudaMemcpyHostToDevice));
    }
}



/// Particule mover (GPU kernel).
/// -----------------------------
__global__ void kernel_mover_PC(particles* part, EMfield* field, grid* grd, parameters* param)
{
    // Index of the particule that is being updated.
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= part->nop)
        return;

    // Auxiliary variables.
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // Local (to the particle) electric and magnetic field.
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // Interpolation densities.
    int ix, iy, iz, idx;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // Intermediate particle position and velocity.
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;


    for (int i_sub=0; i_sub < part->n_sub_cycles; i_sub++) {
        xptilde = part->x[i];
        yptilde = part->y[i];
        zptilde = part->z[i];

        // Calculate the average velocity iteratively.
        for (int innter = 0; innter < part->NiterMover; innter++) {
            // interpolation G-->P
            ix = 2 + int((part->x[i] - grd->xStart)*grd->invdx);
            iy = 2 + int((part->y[i] - grd->yStart)*grd->invdy);
            iz = 2 + int((part->z[i] - grd->zStart)*grd->invdz);

            // calculate weights
            xi[0]   = part->x[i] - grd->XN_flat[get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn)];
            eta[0]  = part->y[i] - grd->YN_flat[get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn)];
            zeta[0] = part->z[i] - grd->ZN_flat[get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn)];

            idx = get_idx(ix, iy, iz, grd->nyn, grd->nzn);
            xi[1]   = grd->XN_flat[idx] - part->x[i];
            eta[1]  = grd->YN_flat[idx] - part->y[i];
            zeta[1] = grd->ZN_flat[idx] - part->z[i];

            // TODO: unroll the loop.
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

            // TODO: unroll the loop.
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for(int kk = 0; kk < 2; kk++) {
                        idx = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);
                        Exl += weight[ii][jj][kk] * field->Ex_flat[idx];
                        Eyl += weight[ii][jj][kk] * field->Ey_flat[idx];
                        Ezl += weight[ii][jj][kk] * field->Ez_flat[idx];
                        Bxl += weight[ii][jj][kk] * field->Bxn_flat[idx];
                        Byl += weight[ii][jj][kk] * field->Byn_flat[idx];
                        Bzl += weight[ii][jj][kk] * field->Bzn_flat[idx];
                    }

            // end interpolation
            omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);
            // solve the position equation
            ut= part->u[i] + qomdt2*Exl;
            vt= part->v[i] + qomdt2*Eyl;
            wt= part->w[i] + qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // solve the velocity equation
            uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
            wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
            // update position
            part->x[i] = xptilde + uptilde*dto2;
            part->y[i] = yptilde + vptilde*dto2;
            part->z[i] = zptilde + wptilde*dto2;
        } // end of iteration

        // update the final position and velocity
        part->u[i]= 2.0*uptilde - part->u[i];
        part->v[i]= 2.0*vptilde - part->v[i];
        part->w[i]= 2.0*wptilde - part->w[i];
        part->x[i] = xptilde + uptilde*dt_sub_cycling;
        part->y[i] = yptilde + vptilde*dt_sub_cycling;
        part->z[i] = zptilde + wptilde*dt_sub_cycling;
       
        // X-DIRECTION: BC particles
        if (part->x[i] > grd->Lx) {
            if (param->PERIODICX) { // PERIODIC
                part->x[i] = part->x[i] - grd->Lx;
            } else { // REFLECTING BC
                part->u[i] = -part->u[i];
                part->x[i] = 2*grd->Lx - part->x[i];
            }
        }

        if (part->x[i] < 0) {
            if (param->PERIODICX) { // PERIODIC
                part->x[i] = part->x[i] + grd->Lx;
            } else { // REFLECTING BC
                part->u[i] = -part->u[i];
                part->x[i] = -part->x[i];
            }
        }

        // Y-DIRECTION: BC particles
        if (part->y[i] > grd->Ly) {
            if (param->PERIODICY) { // PERIODIC
                part->y[i] = part->y[i] - grd->Ly;
            } else { // REFLECTING BC
                part->v[i] = -part->v[i];
                part->y[i] = 2*grd->Ly - part->y[i];
            }
        }

        if (part->y[i] < 0) {
            if (param->PERIODICY) { // PERIODIC
                part->y[i] = part->y[i] + grd->Ly;
            } else { // REFLECTING BC
                part->v[i] = -part->v[i];
                part->y[i] = -part->y[i];
            }
        }

        // Z-DIRECTION: BC particles
        if (part->z[i] > grd->Lz) {
            if (param->PERIODICZ) { // PERIODIC
                part->z[i] = part->z[i] - grd->Lz;
            } else { // REFLECTING BC
                part->w[i] = -part->w[i];
                part->z[i] = 2*grd->Lz - part->z[i];
            }
        }

        if (part->z[i] < 0) {
            if (param->PERIODICZ) { // PERIODIC
                part->z[i] = part->z[i] + grd->Lz;
            } else { // REFLECTING BC
                part->w[i] = -part->w[i];
                part->z[i] = -part->z[i];
            }
        }
    }
}



/// Particle mover (CPU part that launch the GPU kernel).
/// -----------------------------------------------------
int mover_PC(struct particles *part, int is, struct parameters *param)
{
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // Copy particules to GPU.
    CUDA_CHECK(cudaMemcpy(part->GPU_array, part->x, PARRSZ * part->npmax, cudaMemcpyHostToDevice));

    // Move each particle with new fields.
    int num_blocks = (part->nop + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_mover_PC<<<num_blocks, BLOCK_SIZE>>>(&gGpuPart[is], gGpuField, gGpuGrid, gGpuParam);
    cudaDeviceSynchronize();  // Make sure the particules were updated.

DEBUG;

    // Copy particules back to CPU.
    CUDA_CHECK(cudaMemcpy(part->x, part->GPU_array, PARRSZ * part->npmax, cudaMemcpyDeviceToHost));

    return 0;
}



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;

    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    ids->rhon[ix - i][iy - j][iz - k] += weight[i][j][k] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    ids->Jx[ix - i][iy - j][iz - k] += weight[i][j][k] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    ids->Jy[ix - i][iy - j][iz - k] += weight[i][j][k] * grd->invVOL;

        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    ids->Jz[ix - i][iy - j][iz - k] += weight[i][j][k] * grd->invVOL;

        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    ids->pxx[ix - i][iy - j][iz - k] += weight[i][j][k] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    ids->pxy[ix - i][iy - j][iz - k] += weight[i][j][k] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    ids->pxz[ix - i][iy - j][iz - k] += weight[i][j][k] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    ids->pyy[ix - i][iy - j][iz - k] += weight[i][j][k] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    ids->pyz[ix - i][iy - j][iz - k] += weight[i][j][k] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int i=0; i < 2; i++)
            for (int j=0; j < 2; j++)
                for(int k=0; k < 2; k++)
                    ids->pzz[ix -i][iy -j][iz - k] += weight[i][j][k] * grd->invVOL;
    }
}
