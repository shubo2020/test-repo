#ifndef _HELMHOLTZAMRLEVELOP_H_
#define _HELMHOLTZAMRLEVELOP_H_

#include "AMRIO.H"
#include "AMRMultiGrid.H"
#include "BoundaryCondition.H"
#include "CFRegion.H"
#include "CoarseFineInterp.H"
#include "HelmholtzAMRLevelOpF_F.H"
#include "LeastSquareInterpStencil.H"
#include "LevelDataOps.H"
#include "LevelFluxRegister.H"
#include "PrincipalCFInterpStencil.H"
#include "Variable.H"

#include "NamespaceHeader.H"


///
/**
   Operator for solving (alpha I + beta*Laplacian)(phi) = rho
   over an AMR hierarchy.
*/
template<class F=FArrayBox, int Order=4>
class HelmholtzAMRLevelOp  : public AMRLevelOp<LevelData<F> >
{
public:


  template<class T, int Ord> 
  friend class HelmholtzAMRLevelOpFactory;

  typedef BoundaryConditionBase                      BCB;
  typedef RefCountedPtr<BCB>                         BCP;
  //  typedef CoarseFineInterp<LeastSquareInterpStencil> CFI;
  typedef CoarseFineInterp<PrincipalCFInterpStencil> CFI;
  typedef DisjointBoxLayout                          DBL;

  enum{
    isCellAvgd = (TL::IndexOf<typename Variable::CenteringTypes, F>::value
                  ==static_cast<int>(Variable::CellAvgd))
  };

  /**
     \name HelmholtzAMRLevelOp functions */
  /*@{*/

  ///
  /**
   */
  HelmholtzAMRLevelOp() { ; }

  ///
  /**
   */
  virtual ~HelmholtzAMRLevelOp() { ; }

  static const int s_exchangeMode = 0;
  // Point Jacobi : 0; GSRB : 1
  // 2nd-order always use GSRB.
  static const int s_relaxMode = 0; //(Order==2? 1 : 0);
  static const int s_minCoarsestDomainSize = 8;
  static const int s_nGhosts = Order/2;
  //  static const IntVect s_ghostVect = s_nGhosts*IntVect::Unit;
  static const bool enforceConsistency = true;

  ///
  /** define function for AMRLevelOp */
  void define(const DBL& a_grids,
              DBL const*const a_gridsFiner,
              DBL const*const a_gridsCoarser,
              const Real&              a_dxLevel,
              const int                a_refRatio,
              const int                a_refRatioFiner,
              const ProblemDomain&     a_domain,
              const BCP&               a_bc,
              const Copier&            a_exchange,
              const CFRegion&          a_cfregion,
              const int                a_nComps)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::define1");

    m_dx             = a_dxLevel;
    m_refToCoarser   = a_refRatio;
    m_refToFiner     = a_refRatioFiner;
    m_domain         = a_domain;
    m_bc             = a_bc;
    m_exchangeCopier = a_exchange;
    m_cfregion       = a_cfregion;
    m_grids          = a_grids;

    // define flux register if the finer grids exists
    if (a_gridsFiner!=NULL)
      m_levfluxreg.define(*a_gridsFiner, a_grids,
                          refine(m_domain, m_refToFiner),
                          m_refToFiner, a_nComps);

    const bool singleLevel = (a_gridsFiner==NULL && a_gridsCoarser==NULL);
    IntVect tmp = m_refToCoarser*s_minCoarsestDomainSize*IntVect::Unit;
    const bool smallDomain = ( m_domain.domainBox().size() <= tmp );
                               

    // do no bother with coarse-fine interpolation if either is true
    if (singleLevel || smallDomain)
      return;
    
    // parameters for CoarseFineInterp for this use case
    const int  polyDegree = Order;
    // the stencil of discrete Laplacian does not involve corner ghosts.
    const bool fillCorner = false;
    // using proper NestingWidth as 3 to recover Peter's stencil.
    const int  nesting = s_nGhosts;
    const int  homoNesting = s_nGhosts;
    const bool useZeroForHomo = true;
    const bool ghostsOnly = true;
    const bool insistOnAccuracy = false;

    // Two cases for the coarse-fine interpolators
    DBL coarse_grids;
    if (m_refToCoarser==1)
      coarsen(coarse_grids, a_grids, m_refToFiner);
    else
      coarse_grids = *a_gridsCoarser;
    // refToFiner is preferred
    const int r = (m_refToFiner==1? m_refToCoarser : m_refToFiner);
    
    m_homoInterp.define(a_grids, coarse_grids, a_domain, true, polyDegree,
                        r, s_nGhosts, homoNesting, fillCorner,
                        ghostsOnly, insistOnAccuracy, useZeroForHomo);
    m_nonHomoInterp.define(a_grids, coarse_grids, a_domain, false,
                           polyDegree, r, s_nGhosts, nesting, true);
  }

  ///
  /**
     define function for MGLevelOp which has no finer or coarser AMR level
  */
  void define(const DisjointBoxLayout&   a_grids,
              const Real&                a_dx,
              const ProblemDomain&       a_domain,
              const BCP&                 a_bc,
              const Copier&              a_exchange,
              const CFRegion&            a_cfregion)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::define for MGLevelOp");

    m_bc     = a_bc;
    m_domain = a_domain;
    m_dx     = a_dx;
    m_refToCoarser = 2;
    m_refToFiner   = 2;
    m_exchangeCopier = a_exchange;
    m_cfregion = a_cfregion;
    m_grids = a_grids;
  }


  virtual void residual(LevelData<F>&       a_lhs,
                        const LevelData<F>& a_phi,
                        const LevelData<F>& a_rhs,
                        bool a_homogeneous = false)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::residual");
    fillCoarseFineGhostsHomo((LevelData<F>&) a_phi);
    residualI(a_lhs,a_phi,a_rhs,a_homogeneous);
  }


  /// The "I" here means "Ignore the coarse-fine boundary"
  virtual void residualI(LevelData<F>&       a_lhs,
                         const LevelData<F>& a_phi,
                         const LevelData<F>& a_rhs,
                         bool a_homogeneous = false);


  virtual void preCond(LevelData<F>&       a_phi,
                       const LevelData<F>& a_rhs)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::preCond");
    CH_assert(a_phi.nComp() == a_rhs.nComp());

    Real mult = 0;
    // For second-order discretization,
    //  the diagonals of operator L are (alpha - 2*SpaceDim * beta/h/h)
    // so inverse of this is our initial multiplier
    if (Order==2)
      mult = 1.0 / (m_alpha - 2*SpaceDim * m_beta/(m_dx*m_dx) );
    // For fourth-order discretization,
    //  the diagonals of operator L are (alpha - 30*SpaceDim*beta/12/h/h)
    // so inverse of this is our initial multiplier
    else if (Order==4)
      mult = 1.0 / (m_alpha - 30*SpaceDim * m_beta/12.0/(m_dx*m_dx) );

    // don't need to use a Copier -- plain copy will do
    DataIterator dit = a_phi.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
      {
        a_phi[dit].copy(a_rhs[dit]);
        a_phi[dit] *= mult;
      }
    relax(a_phi, a_rhs, 2);
  }


  virtual void applyOp(LevelData<F>&       a_lhs,
                       const LevelData<F>& a_phi,
                       bool a_homogeneous = false)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::applyOp");
    fillCoarseFineGhostsHomo((LevelData<F>&) a_phi);
    applyOpI(a_lhs,a_phi,a_homogeneous);
  }


  /// The "I" here means "Ignore the coarse-fine boundary"
  virtual void applyOpI(LevelData<F>&       a_lhs,
                        const LevelData<F>& a_phi,
                        bool a_homogeneous = false);

  /**
     \name Virtual functions of LinearOp: 
     these cannot be optimized away by inherit from LevelDataOps */
  /*@{*/
  virtual void create(LevelData<F>&       a_lhs,
                      const LevelData<F>& a_rhs)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::create");
    m_levelOps.create(a_lhs, a_rhs);
  }

  virtual void createCoarsened(LevelData<F>&       a_lhs,
                               const LevelData<F>& a_rhs,
                               const int&          a_refRat)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::createCoarsened");

    int ncomp = a_rhs.nComp();
    IntVect ghostVect = a_rhs.ghostVect();
    const DisjointBoxLayout& dbl = a_rhs.disjointBoxLayout();
    CH_assert(dbl.coarsenable(a_refRat));

    DisjointBoxLayout dblCoarsenedFine;
    if(a_refRat == 2)
      {
        if(m_coarsenedMGrids.size() == 0)
          coarsen(m_coarsenedMGrids, dbl, 2);
        dblCoarsenedFine = m_coarsenedMGrids;
      }
    else
      coarsen(dblCoarsenedFine, dbl, a_refRat);

    a_lhs.define(dblCoarsenedFine, ncomp, a_rhs.ghostVect());
  }


  virtual void assign(LevelData<F>&       a_lhs,
                      const LevelData<F>& a_rhs)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::assign");
    m_levelOps.assign(a_lhs, a_rhs);
  }


  virtual void assignLocal(LevelData<F>&       a_lhs,
                           const LevelData<F>& a_rhs)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::assignLocal");
    for (DataIterator dit= a_lhs.dataIterator(); dit.ok(); ++dit)
      a_lhs[dit].copy(a_rhs[dit]);
  }


  virtual void buildCopier(Copier& a_copier,
                           const LevelData<F>& a_lhs,
                           const LevelData<F>& a_rhs)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::buildCopier");
    const DisjointBoxLayout& dbl=a_lhs.disjointBoxLayout();
    a_copier.define(a_rhs.disjointBoxLayout(), dbl, IntVect::Zero);
  }


  virtual void assignCopier(LevelData<F>&       a_lhs,
                            const LevelData<F>& a_rhs,
                            const Copier&       a_copier)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::assignCopier");
    a_rhs.copyTo(a_rhs.interval(), a_lhs, a_lhs.interval(), a_copier);
  }


  virtual void zeroCovered(LevelData<F>& a_lhs,
                           LevelData<F>& a_rhs,
                           const Copier& a_copier)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::zeroCovered");
    m_levelOps.copyToZero(a_lhs, a_copier);
  }


  virtual Real dotProduct(const LevelData<F>& a_1,
                          const LevelData<F>& a_2);


  virtual void incr(LevelData<F>&       a_lhs,
                    const LevelData<F>& a_x,
                    Real                a_scale)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::incr");
    m_levelOps.incr(a_lhs, a_x, a_scale);
  }

  virtual void axby(LevelData<F>&       a_lhs,
                    const LevelData<F>& a_x,
                    const LevelData<F>& a_y,
                    Real                a_a,
                    Real                a_b)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::axby");
    m_levelOps.axby(a_lhs, a_x, a_y, a_a, a_b);
  }


  virtual void scale(LevelData<F>& a_lhs,
                     const Real&   a_scale)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::scale");
    m_levelOps.scale(a_lhs, a_scale);
  }

  /// For linear solvers, using max-norm is good enough.
  virtual Real norm(const LevelData<F>& a_x,
                    int                 a_ord)
  {
    // RelaxSolver defaults norm type to 2.
    //    CH_assert(a_ord==0);
    return localMaxNorm(a_x);
  }

  virtual Real localMaxNorm(const LevelData<F>& a_x);

  virtual void setToZero(LevelData<F>& a_x)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::setToZero");
    m_levelOps.setToZero(a_x);
  }

  /*@}*/

  /**
     \name MGLevelOp functions */
  /*@{*/

  virtual void relax(LevelData<F>&       a_e,
                     const LevelData<F>& a_residual,
                     int                 a_iterations)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::relax");

    // Modify the diagonal weighting for bc's for Point Jacobi
    // if (s_relaxMode==0)
    //   cacheOpDiag(a_e);

    // trivial case of identity operator
    if (m_alpha == 1.0 && m_beta == 0.0)
      {
        a_residual.copyTo(a_e);
        return;
      }

    for (int i = 0; i < a_iterations; i++)
      switch (s_relaxMode)
        {
        case 0:
          cacheOpDiag(a_e);
          levelJacobi(a_e, a_residual);
          break;
        case 1:
          levelGSRB(a_e, a_residual);
          break;
        default:
          MayDay::Abort("unrecognized relaxation mode");
        }
  }


  virtual void createCoarser(LevelData<F>&       a_coarse,
                             const LevelData<F>& a_fine,
                             bool                a_ghosted)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::createCoarser");
    CH_assert(a_fine.disjointBoxLayout().coarsenable(2));
    //multigrid, so coarsen by 2
    if(m_coarsenedMGrids.size() == 0)
      coarsen(m_coarsenedMGrids, a_fine.disjointBoxLayout(), 2); 
    a_coarse.define(m_coarsenedMGrids, a_fine.nComp(), a_fine.ghostVect());
  }

  /**
     calculate restricted residual
     a_resCoarse[2h] = I[h->2h] (rhsFine[h] - L[h](phiFine[h])
  */
  virtual void restrictResidual(LevelData<F>&       a_resCoarse,
                                LevelData<F>& a_phiFine,
                                const LevelData<F>& a_rhsFine);

  /**
     correct the fine solution based on coarse correction
     a_phiThisLevel += I[2h->h](a_correctCoarse)
  */
  virtual void prolongIncrement(LevelData<F>&       a_phi,
                                const LevelData<F>& a_phiCoarse)
  {
    CH_TIME("HelmholtzAMRLevelOp<FArrayBox>::prolongIncrement");
    //this is a multigrid func, hence 2.
    prolong(a_phi, a_phiCoarse, 2); 
  }

  /*@}*/

  /**
     \name AMRLevelOp functions */
  /*@{*/

  /** returns 1 when there are no coarser AMRLevelOp objects */
  virtual int refToCoarser()
  { 
    return m_refToCoarser; 
  }


  /** a_residual = a_rhs - L(a_phi, a_phiFine, a_phiCoarse) */
  virtual void AMRResidual(LevelData<F>&              a_residual,
                           const LevelData<F>&        a_phiFine,
                           const LevelData<F>&        a_phi,
                           const LevelData<F>&        a_phiCoarse,
                           const LevelData<F>&        a_rhs,
                           bool                       a_homogeneousPhysBC,
                           AMRLevelOp<LevelData<F> >* a_finerOp)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::AMRResidual");
    AMROperator(a_residual, a_phiFine, a_phi, a_phiCoarse,
                a_homogeneousPhysBC, a_finerOp);
    axby(a_residual, a_residual, a_rhs, -1.0, 1.0);
  }


  /** residual assuming no more coarser AMR levels */

  virtual void AMRResidualNC(LevelData<F>&              a_residual,
                             const LevelData<F>&        a_phiFine,
                             const LevelData<F>&        a_phi,
                             const LevelData<F>&        a_rhs,
                             bool                       a_homoPhysBC,
                             AMRLevelOp<LevelData<F> >* a_finerOp)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::AMRResidualNC");
    AMROperatorNC(a_residual, a_phiFine, a_phi,
                  a_homoPhysBC, a_finerOp);
    axby(a_residual, a_residual, a_rhs, -1.0, 1.0);
  }


  /** a_residual = a_rhs - L(a_phi, a_phiCoarse)  */
  virtual void AMRResidualNF(LevelData<F>&       a_residual,
                             const LevelData<F>& a_phi,
                             const LevelData<F>& a_phiCoarse,
                             const LevelData<F>& a_rhs,
                             bool                a_homoPhysBC)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::AMRResidualNF");
    fillCoarseFineGhostsNonHomo(a_phi, a_phiCoarse);
    this->residualI(a_residual, a_phi, a_rhs, a_homoPhysBC);
  }


  /**
     a_residual = a_residual - L(a_correction, a_coarseCorrection)
  */
  virtual void AMRUpdateResidual(LevelData<F>&       a_res,
                                 const LevelData<F>& a_corr,
                                 const LevelData<F>& a_crseCorr)
  {
    this->AMRResidualNF(a_res, a_corr, a_crseCorr, a_res, true);
  }


  ///
  /**
     Apply the AMR operator, including coarse-fine matching
  */
  virtual void AMROperator(LevelData<F>&              a_LofPhi,
                           const LevelData<F>&        a_phiFine,
                           const LevelData<F>&        a_phi,
                           const LevelData<F>&        a_phiCoarse,
                           bool                       a_homoPhysBC,
                           AMRLevelOp<LevelData<F> >* a_finerOp)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::AMROperator");
    CH_assert(a_finerOp != NULL);
    fillCoarseFineGhostsNonHomo(a_phi, a_phiCoarse);
    applyOpI(a_LofPhi, a_phi, a_homoPhysBC);
    if (a_phiFine.isDefined())
      reflux(a_phiFine, a_phi, a_LofPhi, a_finerOp);
  }

  ///
  /**
     Apply the AMR operator, including coarse-fine matching
     assume no coarser AMR level
  */
  virtual void AMROperatorNC(LevelData<F>&              a_LofPhi,
                             const LevelData<F>&        a_phiFine,
                             const LevelData<F>&        a_phi,
                             bool                       a_homoPhysBC,
                             AMRLevelOp<LevelData<F> >* a_finerOp)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::AMROperatorNC");
    CH_assert(a_finerOp != NULL);
    applyOpI(a_LofPhi, a_phi, a_homoPhysBC);
    if (a_phiFine.isDefined())
      reflux(a_phiFine, a_phi, a_LofPhi, a_finerOp);
  }


  ///
  /**
     Apply the AMR operator, including coarse-fine matching.
     assume no finer AMR level
  */
  virtual void AMROperatorNF(LevelData<F>&       a_LofPhi,
                             const LevelData<F>& a_phi,
                             const LevelData<F>& a_phiCoarse,
                             bool                a_homoPhysBC)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::AMROperatorNF");
    fillCoarseFineGhostsNonHomo(a_phi, a_phiCoarse);
    this->applyOpI(a_LofPhi, a_phi, a_homoPhysBC);
  }

  /**
     a_resCoarse = I[h-2h]( a_residual - L(a_correction, a_coarseCorrection))
     it is assumed that a_resCoarse has already been filled in with the coarse
     version of AMRResidualNF and that this operation is free to overwrite
     in the overlap regions.
  */
  virtual void AMRRestrict(LevelData<F>&       a_resCrse,
                           const LevelData<F>& a_residual,
                           const LevelData<F>& a_correction,
                           const LevelData<F>& a_crseCorrection,
                           bool a_skip_res = false )
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::AMRRestrict");
    LevelData<F> r;
    create(r, a_residual);
    AMRRestrictS(a_resCrse, a_residual, a_correction, a_crseCorrection,
                 r, a_skip_res);
  }


  virtual void AMRRestrictS(LevelData<F>&       a_resCoarse,
                            const LevelData<F>& a_residual,
                            const LevelData<F>& a_correction,
                            const LevelData<F>& a_coarseCorrection,
                            LevelData<F>&       a_scratch,
                            bool a_skip_res = false )
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::AMRRestrictS");
    if (!a_skip_res)
      AMRResidualNF(a_scratch, a_correction, a_coarseCorrection,
                    a_residual, true);
    else 
      // just copy data (phi in this case, even if its called residual)
      assignLocal(a_scratch, a_residual);

    restrict(a_resCoarse, a_scratch, m_refToCoarser);
  }


  void enforceCFConsistency(LevelData<F>& a_coarseCorrection,
                            const LevelData<F>& a_correction);

  unsigned int orderOfAccuracy(void) const
  {
    return Order;
  }

  /// fill inter-Patch and domain ghost cells
  void fillDomainBdryGhosts(LevelData<F>& phi,
                            const bool a_homo);

  /// fill inter-Patch and domain ghost cells
  void fillNonCoarseFineGhosts(LevelData<F>& phi,
                               const bool a_homo)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::fillNonCoarseFineGhosts");
  
    fillInterPatchGhosts(phi);
    fillDomainBdryGhosts(phi, a_homo);
  }

  /// fill inter-Patch and domain ghost cells
  void fillNonCoarseFineGhosts_FluxBox(LevelData<F>& phi,
                                       const bool a_homo)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::fillNonCoarseFineGhosts");
  
    fillInterPatchGhosts(phi);
    m_bc->fillGhostCells(phi, m_dx, a_homo);
    // const DBL& dbl = phi.disjointBoxLayout();
    // DataIterator dit = phi.dataIterator(); 
    // for (dit.begin(); dit.ok(); ++dit)
    //   for (int d=0; d<SpaceDim; d++)
    //     m_bc(phi[dit][d], dbl[dit], m_domain, m_dx, a_homo);
  }


  /**
     a_correction += I[h->h](a_coarseCorrection)
  */
  virtual void AMRProlong(LevelData<F>&       a_correction,
                          const LevelData<F>& a_coarseCorrection)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,Order>::AMRProlong");

    DBL c;
    coarsen(c, a_correction.disjointBoxLayout(), m_refToCoarser);
    LevelData<F> eCoar(c, a_correction.nComp(), 
                       a_coarseCorrection.ghostVect());
    a_coarseCorrection.copyTo(eCoar.interval(), eCoar, eCoar.interval());
    // when temporary LevelData and copier are passed in,
    //  the above can be replaced with the following line.
    //   a_coarseCorrection.copyTo(a_temp.interval(), a_temp, a_temp.interval(), a_copier);

    // Note the difference of the refRatio from that of Multigrid prolong
    prolong(a_correction, eCoar, m_refToCoarser);
  }


  ///
  /**
     compute norm over all cells on coarse not covered by finer
  */
  virtual Real AMRNorm(const LevelData<F>& a_coarResid,
                       const LevelData<F>& a_fineResid,
                       const int&          a_refRat,
                       const int&          a_ord)
  {
    CH_TIME("HelmholtzAMRLevelOp<F,4>::AMRNorm");

    //create temp and zero out under finer grids
    LevelData<F> coarTemp;
    m_levelOps.create(coarTemp, a_coarResid);
    m_levelOps.assign(coarTemp, a_coarResid);

    if (a_fineResid.isDefined())
      {
        const DisjointBoxLayout& coarGrids = a_coarResid.disjointBoxLayout();
        const DisjointBoxLayout& fineGrids = a_fineResid.disjointBoxLayout();

        int ncomp = coarTemp.nComp();

        for (DataIterator dit = coarGrids.dataIterator(); dit.ok(); ++dit)
          {
            F& coarTempFAB = coarTemp[dit];
            LayoutIterator litFine = fineGrids.layoutIterator();
            for (litFine.reset(); litFine.ok(); ++litFine)
              {
                Box overlayBox = coarTempFAB.box();
                Box coarsenedGrid = coarsen(fineGrids[litFine], a_refRat);
                overlayBox &= coarsenedGrid;
                if (!overlayBox.isEmpty())
                  coarTempFAB.setVal(0.0, overlayBox, 0, ncomp);
              }
          }
      }
    // return norm of temp
    return norm(coarTemp, a_ord);
  }


  /// Change boundary conditions
  virtual void setBC(const BCP& a_bc)
  {
    m_bc = a_bc;
  }


  virtual void reflux(const LevelData<F>&        a_phiFine,
                      const LevelData<F>&        a_phi,
                      LevelData<F>&              a_residual,
                      AMRLevelOp<LevelData<F> >* a_finerOp);

  virtual void getFlux(FluxBox&                    a_flux,
                       const LevelData<F>& a_data,
                       const Box&                  a_grid,
                       const DataIndex&            a_dit,
                       Real                        a_scale);

  virtual void write(const LevelData<F>* a_data,
                     const char*         a_filename);


  /*@}*/

  void setAlphaAndBeta(const double& alpha, const double& beta)
  {
    m_alpha=alpha;
    m_beta =beta;
  }

protected:

  Real                    m_alpha, m_beta;
  Real                    m_dx;
  ProblemDomain           m_domain;
  bool                    m_use2ndOrderSmoother;

  LevelDataOps<F>         m_levelOps;

  LevelData<F>            m_diag;

  BCP                     m_bc;

  CFRegion                m_cfregion;
  Copier                  m_exchangeCopier;

  CFI                     m_nonHomoInterp;
  CFI                     m_homoInterp;

  LevelFluxRegister       m_levfluxreg;

  DisjointBoxLayout       m_grids;
  DisjointBoxLayout       m_coarsenedMGrids;

  int                     m_refToCoarser;
  int                     m_refToFiner;

  virtual void prolong(LevelData<F>&        a_phi,
                       const LevelData<F>&  a_phiCoarse,
                       const int&           a_refRatio);

  virtual void restrict(LevelData<F>&        a_phi,
                        const LevelData<F>&  a_phiFine,
                        const int&           a_refRatio);

  virtual void levelJacobi(LevelData<F>&       a_phi,
                           const LevelData<F>& a_rhs);

  virtual void levelGSRB(LevelData<F>&       a_phi,
                         const LevelData<F>& a_rhs);

  /// figure out the diagonal entry for Jacobi smoothing
  /// In some cases, the diagonal entries depend on the data itself. 
  void cacheOpDiag(const LevelData<F>& a_e);
   
  // In this version of getFlux,
  //  the edgebox is passed in, and the flux array is already defined.
  virtual void getFlux(FArrayBox&       a_flux,
                       const FArrayBox& a_data,
                       const Box&       a_edgebox,
                       int              a_dir,
                       int              a_ref = 1) const
  {
    CH_TIME("HelmholtzAMRLevelOp<FArrayBox,4>::getFlux1");

    CH_assert(isCellAvgd);
    CH_assert(a_dir >= 0);
    CH_assert(a_dir <  SpaceDim);
    CH_assert(!a_data.box().isEmpty());
    // if this fails, the data box was too small (one cell wide, in fact)
    CH_assert(!a_edgebox.isEmpty());
    Real scale = m_beta * a_ref / m_dx;
    if (Order==4)
      {
        FORT_REFLUXGETFLUX4(CHF_FRA(a_flux),
                            CHF_CONST_FRA(a_data),
                            CHF_BOX(a_edgebox),
                            CHF_CONST_REAL(scale),
                            CHF_CONST_INT(a_dir));
      }
    else if (Order==2)
      {
        FORT_REFLUXGETFLUX2(CHF_FRA(a_flux),
                            CHF_CONST_FRA(a_data),
                            CHF_BOX(a_edgebox),
                            CHF_CONST_REAL(scale),
                            CHF_CONST_INT(a_dir));
      }
    else
      {
        MayDay::Error("wrong Order!");
      }
  }


  virtual void getFlux(FArrayBox&       a_flux,
                       const FArrayBox& a_data,
                       int              a_dir,
                       int              a_ref = 1) const
  {
    CH_TIME("HelmholtzAMRLevelOp<FArrayBox,4>::getFlux2");

    Box edgebox = surroundingNodes(a_data.box(),a_dir);
    edgebox.grow(a_dir, -s_nGhosts);
    // if this fails, the data box is too small (one cell wide, in fact)
    CH_assert(!edgebox.isEmpty());
    a_flux.resize(edgebox, a_data.nComp());
    getFlux(a_flux, a_data, edgebox, a_dir, a_ref);
  }


  void fillInterPatchGhosts(LevelData<F>& phi) const
  {
    if (s_exchangeMode == 0)
      phi.exchange(phi.interval());
    else if (s_exchangeMode == 1)
      phi.exchangeNoOverlap(m_exchangeCopier);
    else
      MayDay::Abort("exchangeMode");
  }

  void fillCoarseFineGhostsHomo(LevelData<F>& phi) const;

  void fillCoarseFineGhostsNonHomo
  (const LevelData<F>& a_phi,
   const LevelData<F>& phiCrs) const;

};


#include "NamespaceFooter.H"
#endif
