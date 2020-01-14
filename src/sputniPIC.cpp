/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"


int main(int argc, char **argv)
{
    // Read the inputfile and fill the param structure.
    parameters param;
    readInputFile(&param, argc, argv);
    printParameters(&param);
    saveParameters(&param);

    // Timing variables.
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;

    // Set-up the grid information.
    grid grd;
    setGrid(&param, &grd);

    // Allocate Fields.
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);


    // Allocate Interpolated Quantities per species.
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is = 0; is < param.ns; is++)
        interp_dens_species_allocate(&grd, &ids[is], is);

    // Net densities.
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);

    // Allocate Particles.
    particles *part = new particles[param.ns];
    for (int is = 0; is < param.ns; is++)
        particle_allocate(&param, &part[is], is);

    // Initialization.
    initGEM(&param, &grd, &field, &field_aux, part, ids);
    particle_init_gpu(part, &grd, &param, &field, ids);


    // Count total number of particles.
    long long count = 0;
    for (int is = 0; is < param.ns; is++)
        count += part[is].nop;

    std::cout << "++ TOTAL NUMBER OF PARTICLES: " << count << std::endl;


    // ******************************************************** //
    // **** Start the Simulation!  Cycle index start from 1 *** //
    // ******************************************************** //
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {

        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;

        // Set to zero the densities - needed for interpolation.
        setZeroDensities(&idn,ids,&grd,param.ns);

        // Implicit mover.
        iMover = cpuSecond();  // Start timer for mover.
        for (int is = 0; is < param.ns; is++)
            mover_PC(&part[is], is, &param);
        eMover += (cpuSecond() - iMover);  // Stop timer for mover.


        // Interpolation particle to grid.
        iInterp = cpuSecond();  // Start timer for the interpolation step.

        // Interpolate species.
        for (int is = 0; is < param.ns; is++)
            interpP2G(&part[is], &ids[is], is, &grd, &param);
        // Apply BC to interpolated densities.
        for (int is = 0; is < param.ns; is++)
            applyBCids(&ids[is], &grd, &param);
        // Sum over species.
        sumOverSpecies(&idn, ids, &grd, param.ns);
        // Interpolate charge density from center to node.
        applyBCscalarDensN(idn.rhon, &grd, &param);

        // Write E, B, rho to disk.
        if (cycle % param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }

        eInterp += (cpuSecond() - iInterp);  // Stop timer for interpolation.
    }  // end of one PIC cycle


    // Release the resources, deallocate field.
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    interp_dens_net_deallocate(&grd,&idn);

    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }


    // Print timing of simulation
    double iElaps = cpuSecond() - iStart;

    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles << std::endl;
    std::cout << "**************************************" << std::endl;

    return 0;
}
