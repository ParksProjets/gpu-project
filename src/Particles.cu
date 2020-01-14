#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>


// Check if the given command has returned an error.
#define CUDA_CHECK(cmd) if ((cmd) != cudaSuccess) { \
    printf("ERROR: cuda error at %s:%d\n", __FILE__, __LINE__); abort(); }

#define DEBUG std::cout << "** DEBUG " << __LINE__ << " **" << std::endl


// /!\ W A R N I N G
// The following code assumes that FPpart == FPinterp.
static_assert(sizeof(FPpart) == sizeof(FPinterp));


// Constants.
#define BLOCK_SIZE 32
#define PARRSZ (sizeof(FPpart) * 7)


// Pointer allocated to be used by the CPU.
struct grid *gGpuGrid;
struct parameters *gGpuParam;
struct GPU_interpDensSpecies *gGpuIDS;
struct EMfield *gGpuField;
struct particles *gGpuPart;


// Write a pointer in the CUDA memory.
static void CudaWritePointer(void *dest, const void *src)
{
    CUDA_CHECK(cudaMemcpy(dest, &src, sizeof(void *), cudaMemcpyHostToDevice));
}

// Allocate a structure memeber (array).
static void *CudaAllocateMember(void *dest, long long size)
{
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    CudaWritePointer(dest, ptr);
    return ptr;
}

// Allocate and copy data to the GPU.
static void CudaAllocateAndCopy(void *dest, const void *src, long long size)
{
    void *ptr = CudaAllocateMember(dest, size);
    CUDA_CHECK(cudaMemcpy(ptr, src, size, cudaMemcpyHostToDevice));
}



/// Allocate particle arrays.
/// -------------------------
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    // *Note*: all following parameters are constants.

    part->species_ID = is;  // Set species ID.
    part->nop = param->np[is];  // Number of particles.
    part->npmax = param->npMax[is];  // Maximum number of particles.

    // Choose a different number of mover iterations for ions and electrons.
    if (param->qom[is] < 0) {  // Electrons.
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                   // Ions: only one iteration.
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
    // Thermal.
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];


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



/// Deallocate particle arrays.
/// ---------------------------
void particle_deallocate(struct particles* part)
{
    // We have only one big array for all components of the particles.
    delete[] part->x;
}



/// Initialize GPU data.
/// --------------------
void particle_init_gpu(particles *part, grid *grd, parameters *param, EMfield *field,
    interpDensSpecies *ids)
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


    // Allocate interpDensSpecies array.
    CUDA_CHECK(cudaMalloc(&gGpuIDS, sizeof(GPU_interpDensSpecies) * param->ns));

    for (int is = 0; is < param->ns; is++) {
        ids[is].rhon_GPU = CudaAllocateMember(&gGpuIDS[is].rhon_flat, size);
        ids[is].rhoc_GPU = CudaAllocateMember(&gGpuIDS[is].rhoc_flat, size);
        ids[is].Jx_GPU = CudaAllocateMember(&gGpuIDS[is].Jx_flat, size);
        ids[is].Jy_GPU = CudaAllocateMember(&gGpuIDS[is].Jy_flat, size);
        ids[is].Jz_GPU = CudaAllocateMember(&gGpuIDS[is].Jz_flat, size);
        ids[is].pxx_GPU = CudaAllocateMember(&gGpuIDS[is].pxx_flat, size);
        ids[is].pxy_GPU = CudaAllocateMember(&gGpuIDS[is].pxy_flat, size);
        ids[is].pxz_GPU = CudaAllocateMember(&gGpuIDS[is].pxz_flat, size);
        ids[is].pyy_GPU = CudaAllocateMember(&gGpuIDS[is].pyy_flat, size);
        ids[is].pyz_GPU = CudaAllocateMember(&gGpuIDS[is].pyz_flat, size);
        ids[is].pzz_GPU = CudaAllocateMember(&gGpuIDS[is].pzz_flat, size);
    }


    // Allocate and copy particle array.
    CUDA_CHECK(cudaMalloc(&gGpuPart, sizeof(particles) * param->ns));
    CUDA_CHECK(cudaMemcpy(gGpuPart, part, sizeof(particles) * param->ns, cudaMemcpyHostToDevice));

    // Get memory info: we can use it to know how many particles fit to GPU
    // memory.
    size_t free_size, total_size;
    CUDA_CHECK(cudaMemGetInfo(&free_size, &total_size));

    free_size -= 1024;  // Make sure there is space for aligment.
    size_t maxp = std::min(free_size / PARRSZ, MaxNumberParticules(part, param));
    std::cout << "Max number of particles on GPU = " << maxp << std::endl;

    // Allocate particle array and fill structure with pointers.
    FPpart *array;
    param->gpu_npmax = maxp;
    CUDA_CHECK(cudaMalloc(&array, PARRSZ * maxp));

    for (int is = 0; is < param->ns; is++) {
        part[is].gpu_npmax = std::min(part[is].nop, maxp);
        part[is].GPU_array = array;

        CudaWritePointer(&gGpuPart[is].x, array + (maxp * 0));
        CudaWritePointer(&gGpuPart[is].y, array + (maxp * 1));
        CudaWritePointer(&gGpuPart[is].z, array + (maxp * 2));
        CudaWritePointer(&gGpuPart[is].u, array + (maxp * 3));
        CudaWritePointer(&gGpuPart[is].v, array + (maxp * 4));
        CudaWritePointer(&gGpuPart[is].w, array + (maxp * 5));
        CudaWritePointer(&gGpuPart[is].q, array + (maxp * 6));
    }
}



/// Particle mover (GPU kernel).
/// -----------------------------
__global__ void kernel_mover_PC(long offset, particles* part, EMfield* field, grid* grd, parameters* param)
{
    // Index of the particle that is being updated.
    auto i = blockDim.x * blockIdx.x + threadIdx.x - offset;
    if (i >= part->gpu_npmax)
        return;

    // Auxiliary variables.
    FPpart dt_sub_cycling = (FPpart) param->dt / ((double)part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2 / param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // Local (to the particle) electric and magnetic field.
    FPfield Exl, Eyl, Ezl, Bxl, Byl, Bzl;

    // Interpolation densities.
    int ix, iy, iz, idx;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // Intermediate particle position and velocity.
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
        xptilde = part->x[i];
        yptilde = part->y[i];
        zptilde = part->z[i];

        // Calculate the average velocity iteratively.
        for (int innter = 0; innter < part->NiterMover; innter++) {
            // Interpolation G --> P.
            ix = 2 + int((part->x[i] - grd->xStart) * grd->invdx);
            iy = 2 + int((part->y[i] - grd->yStart) * grd->invdy);
            iz = 2 + int((part->z[i] - grd->zStart) * grd->invdz);

            // Calculate weights.
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

            // Set to zero local electric and magnetic field.
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

            // End interpolation.
            omdtsq = qomdt2 * qomdt2 * (Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0 / (1.0 + omdtsq);
            // Solve the position equation.
            ut = part->u[i] + qomdt2*Exl;
            vt = part->v[i] + qomdt2*Eyl;
            wt = part->w[i] + qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // Solve the velocity equation.
            uptilde = (ut + qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl)) * denom;
            vptilde = (vt + qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl)) * denom;
            wptilde = (wt + qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl)) * denom;
            // Update position.
            part->x[i] = xptilde + uptilde*dto2;
            part->y[i] = yptilde + vptilde*dto2;
            part->z[i] = zptilde + wptilde*dto2;
        } // end of iteration

        // Update the final position and velocity.
        part->u[i] = 2.0*uptilde - part->u[i];
        part->v[i] = 2.0*vptilde - part->v[i];
        part->w[i] = 2.0*wptilde - part->w[i];
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



/// Interpolation Particle --> Grid: This is for species (GPU kernel).
/// ------------------------------------------------------------------
__global__ void kernel_interpP2G(long offset, particles *part, GPU_interpDensSpecies *ids, grid *grd)
{
    // Index of the particle that is being updated.
    auto i = blockDim.x * blockIdx.x + threadIdx.x - offset;
    if (i >= part->gpu_npmax)
        return;

    // Local variables.
    FPpart xi[2], eta[2], zeta[2];
    int ix, iy, iz, idx;
    FPpart value;

    // Determine cell: can we change to int()? is it faster?
    ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
    iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
    iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));

    // Distances from node.
    xi[0]   = part->x[i] - grd->XN_flat[get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn)];
    eta[0]  = part->y[i] - grd->YN_flat[get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn)];
    zeta[0] = part->z[i] - grd->ZN_flat[get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn)];
    idx = get_idx(ix, iy, iz, grd->nyn, grd->nzn);
    xi[1]   = grd->XN_flat[idx] - part->x[i];
    eta[1]  = grd->YN_flat[idx] - part->y[i];
    zeta[1] = grd->ZN_flat[idx] - part->z[i];

    #pragma unroll
    for (int ii = 0; ii < 2; ii++) {
    #pragma unroll
    for (int jj = 0; jj < 2; jj++) {
    #pragma unroll
    for (int kk = 0; kk < 2; kk++) {
        // Calculate the weights for different nodes.
        int weight = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        idx = get_idx(ix - ii, iy - jj, iz - kk, grd->nyn, grd->nzn);

        // Add charge density.
        value = weight * grd->invVOL;
        atomicAdd(&ids->rhon_flat[idx], value);

        // Add current density - Jx.
        value = part->u[i] * weight * grd->invVOL;
        atomicAdd(&ids->Jx_flat[idx], value);

        // Add current density - Jy.
        value = part->v[i] * weight * grd->invVOL;
        atomicAdd(&ids->Jy_flat[idx], value);

        // Add current density - Jz.
        value = part->w[i] * weight * grd->invVOL;
        atomicAdd(&ids->Jz_flat[idx], value);

        // Add pressure pxx.
        value = part->u[i] * part->u[i] * weight * grd->invVOL;
        atomicAdd(&ids->pxx_flat[idx], value);

        // Add pressure pxy.
        value = part->u[i] * part->v[i] * weight * grd->invVOL;
        atomicAdd(&ids->pxy_flat[idx], value);

        // Add pressure pxz.
        value = part->u[i] * part->w[i] * weight * grd->invVOL;
        atomicAdd(&ids->pxz_flat[idx], value);

        // Add pressure pyy.
        value = part->v[i] * part->v[i] * weight * grd->invVOL;
        atomicAdd(&ids->pyy_flat[idx], value);

        // Add pressure pyz.
        value = part->v[i] * part->w[i] * weight * grd->invVOL;
        atomicAdd(&ids->pyz_flat[idx], value);

        // Add pressure pzz.
        value = part->w[i] * part->w[i] * weight * grd->invVOL;
        atomicAdd(&ids->pzz_flat[idx], value);
    } } }
}



/// Copy particules to the GPU for a batch.
/// ---------------------------------------
static void CopyParticlesToDevice(particles *part, parameters *param, long offset)
{
    long maxp = param->gpu_npmax;
    long count = std::min(maxp, part->nop - offset);  // Do not copy to much memory on last batch.
    long size = count * sizeof(FPpart);

    CUDA_CHECK(cudaMemcpy(part->GPU_array + (maxp * 0), part->x + offset, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(part->GPU_array + (maxp * 1), part->y + offset, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(part->GPU_array + (maxp * 2), part->z + offset, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(part->GPU_array + (maxp * 3), part->u + offset, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(part->GPU_array + (maxp * 4), part->v + offset, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(part->GPU_array + (maxp * 5), part->w + offset, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(part->GPU_array + (maxp * 6), part->q + offset, size, cudaMemcpyHostToDevice));
}



/// Copy particules back to the CPU for a batch.
/// --------------------------------------------
static void CopyParticlesToHost(particles *part, parameters *param, long offset)
{
    long maxp = param->gpu_npmax;
    long count = std::min(maxp, part->nop - offset);  // Do not copy to much memory on last batch.
    long size = count * sizeof(FPpart);

    CUDA_CHECK(cudaMemcpy(part->x + offset, part->GPU_array + (maxp * 0), size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->y + offset, part->GPU_array + (maxp * 1), size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->z + offset, part->GPU_array + (maxp * 2), size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->u + offset, part->GPU_array + (maxp * 3), size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->v + offset, part->GPU_array + (maxp * 4), size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->w + offset, part->GPU_array + (maxp * 5), size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part->q + offset, part->GPU_array + (maxp * 6), size, cudaMemcpyDeviceToHost));
}



/// Particle mover (CPU part that launch the GPU kernel).
/// -----------------------------------------------------
void mover_PC(particles *part, int is, parameters *param)
{
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    int num_blocks = (part->gpu_npmax + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Move each particle with new fields (using mini-batches).
    for (long offset = 0; offset < part->nop; offset += part->gpu_npmax) {
        CopyParticlesToDevice(part, param, offset);

        kernel_mover_PC<<<num_blocks, BLOCK_SIZE>>>(offset, &gGpuPart[is], gGpuField, gGpuGrid, gGpuParam);
        cudaDeviceSynchronize();  // Make sure the particles were updated.

        CopyParticlesToHost(part, param, offset);
    }
}



/// Interpolation Particle (CPU part that launch the GPU kernel).
/// -------------------------------------------------------------
void interpP2G(struct particles* part, struct interpDensSpecies* ids, int is, struct grid* grd)
{
    int num_blocks = (part->gpu_npmax + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int size = grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield);

    // Copy interpDensSpecies array to GPU.
    CUDA_CHECK(cudaMemcpy(ids->rhon_GPU, ids->rhon_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids->rhoc_GPU, ids->rhoc_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids->Jx_GPU, ids->Jx_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids->Jy_GPU, ids->Jy_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids->Jz_GPU, ids->Jz_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids->pxx_GPU, ids->pxx_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids->pxy_GPU, ids->pxy_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids->pxz_GPU, ids->pxz_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids->pyy_GPU, ids->pyy_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids->pyz_GPU, ids->pyz_flat, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ids->pzz_GPU, ids->pzz_flat, size, cudaMemcpyHostToDevice));

    // Interpolate each particle (using mini-batches).
    for (long offset = 0; offset < part->nop; offset += part->gpu_npmax) {
        CopyParticlesToDevice(part, param, offset);

        kernel_interpP2G<<<num_blocks, BLOCK_SIZE>>>(offset, &gGpuPart[is], &gGpuIDS[is], gGpuGrid);
        cudaDeviceSynchronize();  // Make sure the particles were updated.

        CopyParticlesToHost(part, param, offset);
    }

    // Copy interpDensSpecies array back to CPU.
    CUDA_CHECK(cudaMemcpy(ids->rhon_flat, ids->rhon_GPU, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids->rhoc_flat, ids->rhoc_GPU, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids->Jx_flat, ids->Jx_GPU, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids->Jy_flat, ids->Jy_GPU, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids->Jz_flat, ids->Jz_GPU, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids->pxx_flat, ids->pxx_GPU, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids->pxy_flat, ids->pxy_GPU, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids->pxz_flat, ids->pxz_GPU, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids->pyy_flat, ids->pyy_GPU, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids->pyz_flat, ids->pyz_GPU, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids->pzz_flat, ids->pzz_GPU, size, cudaMemcpyDeviceToHost))
}
