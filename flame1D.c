
static char help[] = "Nonlinear driven cavity with multigrid in 2d.\n\
  \n\
Keyes and Smooke's combustion model.\n\
Starting procedure eq (3.44)-(3.46) in D. E. Keyes and M. D. Smooke, Flame sheet\n\
The 1D counterflow diffusion flame problem is solved in a velocity-vorticity formulation.\n\
  -guess: initial guess obtained from solution of equations with fixed mu and rho\n\
  -contours : draw contour plots of solution\n\n";
/*How to run:
 * 
 * mpirun -n 4 ./lulu -da_grid_x 100 -snes_monitor_short -contours  -draw_pause -2 -boundS
 * mpirun -n 4 ./lulu -da_grid_x 100 -snes_monitor_short -contours  -draw_pause -2   -boundS -ksp_converged_reason -snes_converged_reason -snes_type aspin -da_overlap 1
 *
 */
/*F-----------------------------------------------------------------------

    This problem is modeled by the partial differential equation system
\begin{eqnarray}
        \frac{dV}{dy} + 2a\rho f' & = & 0  \\
        \frac{d}{dy}(\mu\frac{df'}{dy}) - V\frac{df'}{dy} + a(\rho_inf - \rho(f')^2) & = & 0  \\
        \frac{d}{dy}(\rho D\frac{dS}{dy}) - V\frac{dS}{dy} & = & 0  
\end{eqnarray}

    in [-0.6, 0.8876] (cm), which is uniformly discretized in  x in this simple encoding.
   The boundary conditions at y = -inf given by
                   V = V_minf,
                   f'=\sqrt{\rho_inf/\rho_minf}=f_minf,
                   S = 1,
   and at y = inf by
                   f' = 1,
                   S = 0.

    A finite difference approximation with the usual 3-point stencil
    is used to discretize the boundary value problem to obtain a
    nonlinear system of equations.  Upwinding is used for the divergence
    (convective) terms and central for the gradient (source) terms.

    The Jacobian can be either
      * formed via finite differencing using coloring (the default), or
      * applied matrix-free via the option -snes_mf
        (for larger grid problems this variant may not converge
        without a preconditioner due to ill-conditioning).

  ------------------------------------------------------------------------F*/
#if defined(PETSC_APPLE_FRAMEWORK)
#import <PETSc/petscsnes.h>
#import <PETSc/petscdmda.h>
#else
#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petsc/private/snesimpl.h> 

#endif

/*
   User-defined routines and data structures
*/
typedef struct {
  PetscScalar V,fprime,S;
} Field;

typedef struct {
  PetscBool   draw_contours;    /* flag - 1 indicates drawing contours */
  PetscBool   guess;            /* flag - 1 indicates drawing contours */
  PetscBool   boundS;           /* flag - 1 indicates bounding S for SNES solver */
  PetscBool   pseudo_time;      /* flag -1 indicates using pseudo time term */
  
  PetscScalar L_F;              /* boundary at the fuel jet */
  PetscScalar L_OX;             /* boundary at the oxidizer jet */
  PetscScalar L;                /* The separation distance of the jets */
  PetscScalar a;                /* strain rate; */
  PetscScalar f_minf;           /* boundary condition for streamfunction, f_{-inf}*/ 
  PetscScalar V_minf;           /* boundary condition for velocity V_{-inf} */
  PetscScalar Prandtl;          /* ratio of thermal and mass diffusities */
  PetscScalar rho_inf;          /* density at infinity */
  PetscScalar mu_0,T_0;         /* reference values for air */  
  PetscScalar T_inf,T_minf,Tb;     /* temperature at inf and -inf, burnt temprature */
  PetscScalar delta_T;          /* per mass heat release divided by heat capacity */
  
  PetscScalar W_F;              /* Molecular mass of fuel (methan) */
  PetscScalar W_O;              /* Molecular mass of oxidizer (oxygen) */     
  PetscScalar W_P;              /* Molecular mass of product (carbon dioxide and two waters)*/
  PetscScalar W_N;              /* Molecular mass of inert (essentially, nitrogen) */
  PetscScalar nu_F;             /* stoichiometric coefficient of fuel */
  PetscScalar nu_O;             /* stoichiometric coefficient of oxidizer */
  PetscScalar nu_P;             /* stoichiometric coefficient of product */
  PetscScalar Y_O_inf;          /* mass fraction of oxidizer at inf */
  PetscScalar Y_F_minf;         /* mass fraction of fuel at -inf */
  PetscScalar Y_N_inf;          /* mass fraction of inert at inf */
  PetscScalar Y_N_minf;         /* mass fraction of inert at -inf */

  PetscScalar W_inf;            /* molecular mass at inf */
  PetscScalar S_f;              /* level set value for flamefront in S(x) */  
  PetscScalar gas_const;     /* p/R ratio in gas constant, evaluated at inf */

  Vec         Xold1,Xold2,Xlow,Xup;
  PetscScalar theta,epsilon;
  PetscScalar dt,dt_prev1, dt_prev2;
  PetscScalar dt_max,dt_min;

} AppCtx;
PetscErrorCode Monitor(SNES,PetscInt,PetscReal,void*);
PetscErrorCode FormFunctionLocal_guess(DMDALocalInfo*,Field*,Field*,void*);
PetscErrorCode FlameFront(PetscScalar,PetscScalar*,PetscScalar*,void*);
static PetscErrorCode CoefficientSubDomainRestrictHook(DM,DM,void*);
static PetscErrorCode FormOldSolution(DM, Vec);

PetscErrorCode FormFunctionLocal(DMDALocalInfo*,Field*,Field*,void*);
PetscErrorCode FormInitialGuess(AppCtx*,DM,Vec);

/*--------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(SNES snes,PetscInt its,PetscReal fnorm,void *ctx)
{
  AppCtx      *user = (AppCtx*) ctx;
  Vec         X,F;
  PetscErrorCode ierr;
  //SNESSetConvergenceTest  
  ierr = VecLockPop(snes->vec_sol);CHKERRQ(ierr);

  ierr = SNESGetSolution(snes,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);  
  ierr = VecPointwiseMax(X,X,user->Xlow);CHKERRQ(ierr);
  ierr = VecPointwiseMin(X,X,user->Xup);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F); CHKERRQ(ierr);

  ierr = VecLockPush(snes->vec_sol);CHKERRQ(ierr);

  return 0;
}    


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,its;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DM             da,Sda;
  Vec            X,W,Y1,Y2,s;
  PetscInt       step;
  PetscScalar    dt,norm,ftime=0.0;
  PetscScalar    theta,epsilon,tol,factor_dt,alpha;
  SNESConvergedReason reason;


  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return(1);

  PetscFunctionBeginUser;
  comm = PETSC_COMM_WORLD;

  /*
      Create distributed array object to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
  */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,3,1,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);  
  
  ierr = DMDASetUniformCoordinates(da, -0.6, 0.8876, 0.0, 1.0, 0.0, 1.0);  

  ierr = DMDAGetInfo(da,0,&mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  /*
     Problem parameters
  */
  user.dt = 1.0;
  user.dt_max = 10;
  user.dt_min = 1.e-7;
  factor_dt = 1.5;
  theta   = 0.75;
//  epsilon = 1.0e-01;
  tol = 1.0-2; /* deltaX<tol */

  user.L_F = -0.6;
  user.L_OX = 0.8876;
  user.L = user.L_OX-user.L_F;
  user.a = 40.0;
  user.f_minf = 1.216;
  user.V_minf = 0.028;
  user.Prandtl = 0.75;
  user.rho_inf = 0.00124;
  user.mu_0 = 0.000185;
  user.T_0 = 298;
  user.T_inf = 294;
  user.T_minf = 294;
  user.Tb = 1400+294;
  user.W_F = 16;
  user.W_O = 32;
  user.W_P = 80.0/3.0;
  user.W_N = 28;
  user.nu_F = 1;
  user.nu_O = 2;
  user.nu_P = 3;
  user.Y_O_inf = 0.18;
  user.Y_F_minf = 0.598;
  user.Y_N_inf = 1-user.Y_O_inf;
  user.Y_N_minf = 1-user.Y_F_minf;
  user.W_inf = user.Y_O_inf *user.W_O+user.Y_N_inf*user.W_N;
  user.S_f = 1./(1. + (user.W_O*user.nu_O*user.Y_F_minf)/(user.W_F*user.nu_F*user.Y_O_inf));
  user.gas_const = user.rho_inf*user.T_inf/user.W_inf;

  user.delta_T = ((user.Tb - user.T_inf*(1-user.S_f))/user.S_f-user.T_minf)/user.Y_F_minf;


  PetscPrintf(comm, "W_inf #= %g,  S_f #=%g, gas_const #=%g\n",user.W_inf,user.S_f,user.gas_const); 


  ierr = PetscOptionsHasName(NULL,NULL,"-contours",&user.draw_contours);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-guess",&user.guess);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-boundS",&user.boundS);CHKERRQ(ierr);
  
//  user.guess = PETSC_TRUE;

  ierr = DMDASetFieldName(da,0,"V");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"fprime");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,2,"S");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     Also, compute the initial guess.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&X);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&Y1);CHKERRQ(ierr); 
  ierr = DMCreateGlobalVector(da,&Y2);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&W);CHKERRQ(ierr);    
  ierr = DMCreateGlobalVector(da,&user.Xold1);CHKERRQ(ierr); 
  ierr = DMCreateGlobalVector(da,&user.Xold2);CHKERRQ(ierr);    
  ierr = DMCreateGlobalVector(da,&user.Xlow);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&user.Xup);CHKERRQ(ierr);

  /* Set lower and upper bound */
  ierr = DMDAGetReducedDMDA(da,1,&Sda);CHKERRQ(ierr);    
  ierr = DMCreateGlobalVector(Sda,&s);CHKERRQ(ierr);  
  ierr = VecSet(user.Xlow,-1.e+15);CHKERRQ(ierr);
  ierr = VecSet(s,0.0);CHKERRQ(ierr);
  ierr = VecStrideScatter(s,2,user.Xlow,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSet(user.Xup,1.e+15);CHKERRQ(ierr);
  ierr = VecSet(s,1.0);CHKERRQ(ierr);
  ierr = VecStrideScatter(s,2,user.Xup,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecDestroy(&s);CHKERRQ(ierr);  


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = FormInitialGuess(&user,da,X);CHKERRQ(ierr);
  user.dt_prev1 = user.dt;
  user.dt_prev2 = user.dt;
  VecCopy(X,user.Xold1);
  VecCopy(X,user.Xold2);

  user.pseudo_time=PETSC_TRUE;
  step=0;
  while(step<1000){
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%d TS dt %g time %g\n",step,user.dt,ftime);CHKERRQ(ierr); 
//  if(step==5){user.pseudo_time=PETSC_FALSE;step=-100;}

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,3,1,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);  
  
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);        
  ierr = FormOldSolution(da, user.Xold1);CHKERRQ(ierr);

  ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,(DM)da);CHKERRQ(ierr); 
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user);CHKERRQ(ierr);  
  if(user.boundS){ierr = SNESMonitorSet(snes,Monitor,&user,0);CHKERRQ(ierr);}  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  

  if(reason>0) {

      ftime = ftime +user.dt;        
      step = step +1;

      if(step>=2)user.dt_prev2 = user.dt_prev1; 
      user.dt_prev1  = user.dt;
          
      if(step>=2){

          VecWAXPY(Y1,-1.0,user.Xold1,X);
          VecNorm(Y1,NORM_2,&norm);
          norm=norm*PetscSqrtScalar(1.0/mx);
          PetscPrintf(PETSC_COMM_WORLD,"---------------||X-Xn||/N=%g----------------\n",norm);
/*          if(norm<tol){
              PetscPrintf(PETSC_COMM_WORLD,"------ The error||X-Xn||/N=%g-------\n",norm);
              break;
          }*/

          if(user.dt>user.dt_max){
              PetscPrintf(PETSC_COMM_WORLD,"------ dt=%g > dt_max=%g-------\n",user.dt,user.dt_max);              
              break;
          }

          /* alpha = max((X_n-X_{n-1})/dt_n-(X_{n-1}-X_{n-2})/dt_{n-1})*/
          VecScale(Y1,1.0/user.dt_prev1);
          VecWAXPY(Y2,-1.0,user.Xold2,user.Xold1);
          VecScale(Y2,1.0/user.dt_prev2);
          VecAXPY(Y1,-1.0,Y2);
          VecAbs(Y1);
          VecMax(Y1,NULL,&alpha);


          VecCopy(X,Y2);
          VecAbs(Y2);
          VecMax(Y2,NULL,&epsilon); 
          dt = factor_dt*user.dt;             
          user.dt = PetscSqrtScalar((user.dt_prev1+user.dt_prev2)*(epsilon+1.)*theta/alpha);
          PetscPrintf(PETSC_COMM_WORLD,"dt =====%g,   user.dt=======%g    epsilon=%g\n",dt,user.dt,epsilon);       
          user.dt = PetscMin(dt,user.dt);
          VecCopy(user.Xold2,user.Xold1); 
      }
      VecCopy(X,user.Xold1);      

    }else{
          ierr = VecCopy(user.Xold1,X);CHKERRQ(ierr);
          PetscPrintf(PETSC_COMM_WORLD,"# SNES fails and cut time step from %g into %g\n",user.dt,0.5*user.dt);
          user.dt=0.5*user.dt;
     }

     if(user.dt<user.dt_min){
         PetscPrintf(PETSC_COMM_WORLD,"# Please give a better initial guess!\n");
     }


  if (user.draw_contours) {
    ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }

  
}

  PetscPrintf(PETSC_COMM_WORLD,"\n-----------------Full Newton Method---------------\n");
  user.pseudo_time=PETSC_FALSE;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,3,1,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);  
  
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);        
  ierr = FormOldSolution(da, user.Xold1);CHKERRQ(ierr);
  ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,(DM)da);CHKERRQ(ierr); 
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user);CHKERRQ(ierr);  
  if(user.boundS){ierr = SNESMonitorSet(snes,Monitor,&user,0);CHKERRQ(ierr);}  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);

  if (user.draw_contours) {
    ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }



   /* Write new vector in binary format */

  PetscViewer    viewer;
  PetscLogEvent  VECTOR_WRITE;

  ierr = PetscLogEventRegister("Write Vector",VEC_CLASSID,&VECTOR_WRITE);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(VECTOR_WRITE,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"writting vector in binary to solution.dat ...\n");CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"solution_mod.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(X,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VECTOR_WRITE,0,0,0,0);CHKERRQ(ierr);




  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y1);CHKERRQ(ierr);
  ierr = VecDestroy(&Y2);CHKERRQ(ierr);
  
//  ierr = DMDestroy(&da);CHKERRQ(ierr);
//  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

/* ------------------------------------------------------------------- */


#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
*/
PetscErrorCode FormInitialGuess(AppCtx *user,DM da,Vec X)
{
  PetscInt       i,mx,xs,xm;
  PetscErrorCode ierr;
  PetscReal      dx,L,f_minf,V_minf;
  Field          *x;

  L = user->L;
  V_minf = user->V_minf;
  f_minf = user->f_minf;

  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx   = L/(mx-1);

  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs   - starting grid indices (no ghost points)
       xm   - widths of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
  */
    for (i=xs; i<xs+xm; i++) {
      x[i].V = V_minf;
      x[i].fprime = (1-f_minf)/L*i*dx + f_minf;
      x[i].S = -1./L*i*dx + 1.;

/*      x[i].V=0.0;
      x[i].fprime=0.0;
      x[i].S=0.0;*/
     }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);

  if(user->guess){
      SNES           snes1;
      DM             da1;
      Vec            Y;
      ierr = SNESCreate(PETSC_COMM_WORLD,&snes1);CHKERRQ(ierr);
      ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,3,1,NULL,&da1);CHKERRQ(ierr);
      ierr = DMSetFromOptions(da1);CHKERRQ(ierr);
      ierr = DMSetUp(da1);CHKERRQ(ierr);  
	  
      ierr = DMDASetUniformCoordinates(da1, -0.6, 0.8876, 0.0, 0.1, 0.0, 0.1);     
      ierr = SNESSetDM(snes1,(DM)da1);CHKERRQ(ierr);

      ierr = DMDASetFieldName(da1,0,"V");CHKERRQ(ierr);
      ierr = DMDASetFieldName(da1,1,"fprime");CHKERRQ(ierr);
      ierr = DMDASetFieldName(da1,2,"S");CHKERRQ(ierr);


      ierr = SNESSetOptionsPrefix(snes1,"initial_guess_");CHKERRQ(ierr);      
      ierr = DMDASNESSetFunctionLocal(da1,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal_guess,user);CHKERRQ(ierr);
      ierr = SNESSetFromOptions(snes1);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(da1,&Y);CHKERRQ(ierr);  
      ierr = VecCopy(X,Y);  
      ierr = SNESSolve(snes1,NULL,Y);CHKERRQ(ierr);
      ierr = VecCopy(Y,X);
//      ierr = VecView(Y,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
}


  return 0;
}




#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,Field *x,Field *f,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       xints,xinte,i;
  PetscReal      hx,dhx;
  PetscReal      a,Prandtl,L,rho_inf,V_minf,f_minf,gas_const;
  PetscReal      T_0,mu_0;
  PetscReal      W,T,rho,mu;
  PetscReal      WL,WR,TL,TR,mu_L,mu_R;
  PetscScalar    u,uxx,vx,avx,vxp,vxm;

  PetscScalar    udot;
  Field          *xold;
  Vec            Xold;
  DM             cdm;

  PetscFunctionBeginUser;
  a = user->a;
  L = user->L;
  Prandtl = user->Prandtl;
  rho_inf = user->rho_inf;
  V_minf = user->V_minf;
  f_minf = user->f_minf;
  gas_const = user->gas_const;
  mu_0 = user->mu_0;
  T_0 = user->T_0;

  ierr = PetscObjectQuery((PetscObject)info->da,"coefficientdm",(PetscObject*)&cdm);CHKERRQ(ierr);
  ierr = DMGetNamedLocalVector(cdm,"coefficient",&Xold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cdm,Xold,&xold);CHKERRQ(ierr);


  /*
     Define mesh intervals ratios for uniform grid.

     Note: FD formulae below are normalized by multiplying through by
     local volume element (i.e. hx) to obtain coefficients O(1) in one dimensions.


  */
  dhx   = (PetscReal)(info->mx-1)/L;    
  hx    = 1.0/dhx;                  
  xints = info->xs; xinte = info->xs+info->xm; 


  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    f[i].V = x[i].V - V_minf;
    f[i].fprime = x[i].fprime - f_minf;
    f[i].S = x[i].S - 1.0;

//    PetscPrintf(PETSC_COMM_WORLD,"v=%g   fp=%g   S=%g\n",f[i].V, f[i].fprime, f[i].S);
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i     = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */
    ierr = FlameFront(x[i].S,&T,&W,user);CHKERRQ(ierr);
    rho = gas_const*W/T;
    
    f[i].V     = (x[i].V - x[i-1].V)*dhx + 2.0*a*rho*x[i].fprime;
    f[i].fprime = x[i].fprime - 1.0;
    f[i].S = x[i].S;
    
  }

  /* Compute over the interior points */

  for (i=xints; i<xinte; i++) {


      ierr = FlameFront(x[i].S,&T,&W,user);CHKERRQ(ierr);
      ierr = FlameFront(x[i-1].S,&TL,&WL,user);CHKERRQ(ierr);
      ierr = FlameFront(x[i+1].S,&TR,&WR,user);CHKERRQ(ierr);

      rho = gas_const*W/T;

      mu = mu_0*PetscPowScalar(T/T_0,0.7);
      mu_L = mu_0*PetscPowScalar(TL/T_0,0.7);
      mu_R = mu_0*PetscPowScalar(TR/T_0,0.7);


      /*
       convective coefficients for upwinding
      */
      vx  = x[i].V; avx = PetscAbsScalar(vx);
      vxp = .5*(vx+avx); vxm = .5*(vx-avx);

      /* V */
      f[i].V = (x[i].V - x[i-1].V)*dhx + 2.0*a*rho*x[i].fprime;
         
      /* fprime */
      u = x[i].fprime;
      uxx = 0.5*(mu + mu_R)*(x[i+1].fprime-u)
            - 0.5*(mu + mu_L)*(u-x[i-1].fprime);
      uxx = uxx*dhx*dhx;      
      f[i].fprime = uxx - (PetscMax(0.5*(x[i].V+x[i-1].V),0)*(u - x[i-1].fprime) - PetscMax(-0.5*(x[i].V+x[i+1].V),0)*(x[i+1].fprime -u))*dhx
                    + a *(rho_inf - rho*x[i].fprime*x[i].fprime);

      udot      = user->pseudo_time ? (x[i].fprime-xold[i].fprime)/user->dt : 0.;
      f[i].fprime = -udot+f[i].fprime;

              

      /* S */
      u = x[i].S;
      uxx = 0.5*(mu + mu_R)*(x[i+1].S-u)
            - 0.5*(mu + mu_L)*(u-x[i-1].S);
      uxx = uxx*dhx*dhx;      
      f[i].S = uxx/Prandtl - (PetscMax(0.5*(x[i].V+x[i-1].V),0)*(u - x[i-1].S) - PetscMax(-0.5*(x[i].V+x[i+1].V),0)*(x[i+1].S -u))*dhx;

      udot      = user->pseudo_time ? (x[i].S-xold[i].S)/user->dt : 0.;
      f[i].S = -udot+f[i].S;
      
       
    }


  ierr = DMDAVecRestoreArray(cdm,Xold,&xold);CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(cdm,"coefficient",&Xold);CHKERRQ(ierr);
    

  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "FlameFront"
PetscErrorCode FlameFront(PetscScalar S,PetscScalar *T,PetscScalar *W,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscReal      S_f;
  PetscReal      T_inf,T_minf,delta_T;
  PetscReal      W_F,W_O,W_P,W_N,nu_F,nu_O,nu_P;
  PetscReal      Y_O_inf,Y_F_minf,Y_N_inf,Y_N_minf;
  PetscReal      Y_F,Y_O,Y_P,Y_N;

  PetscFunctionBeginUser;
  S_f = user->S_f;
  T_inf = user->T_inf;
  T_minf = user->T_minf;
  delta_T = user->delta_T;
  W_F = user->W_F;
  W_O = user->W_O;
  W_P = user->W_P;
  W_N = user->W_N;
  nu_F = user->nu_F;
  nu_O = user->nu_O;
  nu_P = user->nu_P;
  Y_O_inf = user->Y_O_inf;
  Y_F_minf = user->Y_F_minf;
  Y_N_inf = user->Y_N_inf;
  Y_N_minf = user->Y_N_minf;

  if (S>=S_f){
      *T = T_minf*S + (T_inf +delta_T*(W_F*nu_F)/(W_O*nu_O)*Y_O_inf)*(1-S);
      Y_F = Y_F_minf*S + Y_O_inf*(W_F*nu_F)/(W_O*nu_O)*(S - 1.0);
      Y_O = 0.0;
      Y_P = Y_O_inf*(W_P*nu_P)/(W_O*nu_O)*(1-S);
      Y_N = Y_N_inf*(1.0-S) + Y_N_minf*S;
      *W = 1.0/(Y_F/W_F + Y_P/W_P + Y_N/W_N);
      
   }else{
      *T = T_inf*(1.0-S) + (T_minf + delta_T*Y_F_minf)*S;
      Y_F = 0.0;
      Y_O = Y_O_inf*(1.0-S) - Y_F_minf*(W_O*nu_O)/(W_F*nu_F)*S;
      Y_P = Y_F_minf*(W_P*nu_P)/(W_F*nu_F)*S;
      Y_N = Y_N_inf*(1.0-S) + Y_N_minf*S;
      *W = 1.0/(Y_O/W_O + Y_P/W_P + Y_N/W_N);
      
   }


  PetscFunctionReturn(0);

}    

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal_guess"
PetscErrorCode FormFunctionLocal_guess(DMDALocalInfo *info,Field *x,Field *f,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
//  PetscErrorCode ierr;
  PetscInt       xints,xinte,i;
  PetscReal      hx,dhx;
  PetscReal      a,Prandtl,L,rho_inf,V_minf,f_minf;
  PetscReal      mu_0;
  PetscReal      rho,mu;
  PetscScalar    u,uxx,vx,avx,vxp,vxm;


  PetscFunctionBeginUser;
  a = user->a;
  L = user->L;
  Prandtl = user->Prandtl;
  rho_inf = user->rho_inf;
  V_minf = user->V_minf;
  f_minf = user->f_minf;
  mu_0 = user->mu_0;

  rho=rho_inf;      
  mu=mu_0;

  /*
     Define mesh intervals ratios for uniform grid.

     Note: FD formulae below are normalized by multiplying through by
     local volume element (i.e. hx) to obtain coefficients O(1) in one dimensions.


  */
  dhx   = (PetscReal)(info->mx-1)/L;    
  hx    = 1.0/dhx;                  
  xints = info->xs; xinte = info->xs+info->xm; 

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    f[i].V = x[i].V - V_minf;
    f[i].fprime = x[i].fprime - f_minf;
    f[i].S = x[i].S - 1.0;
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i     = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */
    
    f[i].V     = (x[i].V - x[i-1].V)*dhx + 2.0*a*rho*x[i].fprime;
    f[i].fprime = x[i].fprime - 1.0;
    f[i].S = x[i].S;
  }

  /* Compute over the interior points */

  for (i=xints; i<xinte; i++) {
      /*
       convective coefficients for upwinding
      */
      vx  = x[i].V; avx = PetscAbsScalar(vx);
      vxp = .5*(vx+avx); vxm = .5*(vx-avx);

      /* V */
      f[i].V = (x[i].V - x[i-1].V)*dhx + 2.0*a*rho*x[i].fprime;
      
      /* fprime */
      u = x[i].fprime;
      uxx = mu*(x[i+1].fprime + x[i-1].fprime -2.0*u);
      uxx = uxx*dhx;      
      f[i].fprime = uxx - (PetscMax(0.5*(x[i].V+x[i-1].V),0)*(u - x[i-1].fprime) - PetscMax(-0.5*(x[i].V+x[i+1].V),0)*(x[i+1].fprime -u))
                    + a *(rho_inf - rho*x[i].fprime*x[i].fprime)*hx;

      /* S */
      u = x[i].S;
      uxx = mu*(x[i+1].S + x[i-1].S -2.0*u);
      uxx = uxx*dhx;      
      f[i].S = uxx/Prandtl - (PetscMax(0.5*(x[i].V+x[i-1].V),0)*(u - x[i-1].S) - PetscMax(-0.5*(x[i].V+x[i+1].V),0)*(x[i+1].S -u));
       
    }

  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "FormOldSolution"
  /* set up the coefficient 
 */
static PetscErrorCode FormOldSolution(DM da,Vec coeff)
{
  DM             cda;
  Vec            c,clocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMDAGetReducedDMDA(da,3,&cda);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)da,"coefficientdm",(PetscObject)cda);CHKERRQ(ierr);

  ierr = DMGetNamedGlobalVector(cda,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMGetNamedLocalVector(cda,"coefficient",&clocal);CHKERRQ(ierr);

  ierr = VecCopy(coeff,c);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(cda,c,INSERT_VALUES,clocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(cda,c,INSERT_VALUES,clocal);CHKERRQ(ierr);

  ierr = DMRestoreNamedLocalVector(cda,"coefficient",&clocal);CHKERRQ(ierr);
  ierr = DMRestoreNamedGlobalVector(cda,"coefficient",&c);CHKERRQ(ierr);

  ierr = DMSubDomainHookAdd(da,CoefficientSubDomainRestrictHook,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "CoefficientSubDomainRestrictHook"
/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode CoefficientSubDomainRestrictHook(DM dm,DM subdm,void *ctx)
{
  Vec            c,cc;
  DM             cdm,csubdm;
  PetscErrorCode ierr;
  VecScatter     *iscat,*oscat,*gscat;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dm,"coefficientdm",(PetscObject*)&cdm);CHKERRQ(ierr);

  if (!cdm) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The coefficient DM needs to be set up!");

  ierr = DMDAGetReducedDMDA(subdm,3,&csubdm);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)subdm,"coefficientdm",(PetscObject)csubdm);CHKERRQ(ierr);

  ierr = DMGetNamedGlobalVector(cdm,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMGetNamedLocalVector(csubdm,"coefficient",&cc);CHKERRQ(ierr);

  ierr = DMCreateDomainDecompositionScatters(cdm,1,&csubdm,&iscat,&oscat,&gscat);CHKERRQ(ierr);

  ierr = VecScatterBegin(*gscat,c,cc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(*gscat,c,cc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecScatterDestroy(iscat);CHKERRQ(ierr);
  ierr = VecScatterDestroy(oscat);CHKERRQ(ierr);
  ierr = VecScatterDestroy(gscat);CHKERRQ(ierr);
  ierr = PetscFree(iscat);CHKERRQ(ierr);
  ierr = PetscFree(oscat);CHKERRQ(ierr);
  ierr = PetscFree(gscat);CHKERRQ(ierr);

  ierr = DMRestoreNamedGlobalVector(cdm,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(csubdm,"coefficient",&cc);CHKERRQ(ierr);

  ierr = DMDestroy(&csubdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

