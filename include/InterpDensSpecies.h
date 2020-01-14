#ifndef INTERPDENSSPECIES_H
#define INTERPDENSSPECIES_H

#include "Alloc.h"
#include "PrecisionTypes.h"
#include "Grid.h"

/** Interpolated densities per species on nodes */
struct interpDensSpecies {
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    // index 1: rho
    FPinterp*** rhon; FPinterp *rhon_flat; void *rhon_GPU;
    FPinterp*** rhoc; FPinterp *rhoc_flat; void *rhoc_GPU;
    
    // index 2, 3, 4
    FPinterp*** Jx; FPinterp *Jx_flat; void *Jx_GPU;
    FPinterp*** Jy; FPinterp *Jy_flat; void *Jy_GPU;
    FPinterp*** Jz; FPinterp *Jz_flat; void *Jz_GPU;
    // index 5, 6, 7, 8, 9, 10: pressure tensor (symmetric)
    FPinterp*** pxx; FPinterp *pxx_flat; void *pxx_GPU;
    FPinterp*** pxy; FPinterp *pxy_flat; void *pxy_GPU;
    FPinterp*** pxz; FPinterp *pxz_flat; void *pxz_GPU;
    FPinterp*** pyy; FPinterp *pyy_flat; void *pyy_GPU;
    FPinterp*** pyz; FPinterp *pyz_flat; void *pyz_GPU;
    FPinterp*** pzz; FPinterp *pzz_flat; void *pzz_GPU;
};


/** Interpolated densities per species on nodes (GPU data). */
struct GPU_interpDensSpecies {
    FPinterp *rhon_flat;
    FPinterp *rhoc_flat;
    FPinterp *Jx_flat;
    FPinterp *Jy_flat;
    FPinterp *Jz_flat;
    FPinterp *pxx_flat;
    FPinterp *pxy_flat;
    FPinterp *pxz_flat;
    FPinterp *pyy_flat;
    FPinterp *pyz_flat;
    FPinterp *pzz_flat;
};

/** allocated interpolated densities per species */
void interp_dens_species_allocate(struct grid* grd, struct interpDensSpecies* ids, int is);

/** deallocate interpolated densities per species */
void interp_dens_species_deallocate(struct grid* grd, struct interpDensSpecies* ids);

/** deallocate interpolated densities per species */
void interpN2Crho(struct interpDensSpecies* ids, struct grid* grd);

#endif
