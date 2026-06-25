//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  mat_gen_fd_dh.c
 *  Date: 2013/01/23
 */

#include "jx_euclid.h"

static jx_bool jx_isThreeD;

#define JX_FRONT(a) a[5]
#define JX_SOUTH(a) a[3]
#define JX_WEST(a) a[1]
#define JX_CENTER(a) a[0]
#define JX_EAST(a) a[2]
#define JX_NORTH(a) a[4]
#define JX_BACK(a) a[6]
#define JX_RHS(a) a[7]

static void jx_setBoundary_private( JX_Int node, JX_Int *cval, JX_Real *aval,
                 JX_Int len, JX_Real *rhs, JX_Real bc, JX_Real coeff, JX_Real ctr, JX_Int nabor );
static void jx_generateStriped( jx_MatGenFD mg, JX_Int *rp, JX_Int *cval, JX_Real *aval, jx_Mat_dh A, jx_Vec_dh b );
static void jx_generateBlocked( jx_MatGenFD mg, JX_Int *rp, JX_Int *cval, JX_Real *aval, jx_Mat_dh A, jx_Vec_dh b );
static void jx_getstencil( jx_MatGenFD g, JX_Int ix, JX_Int iy, JX_Int iz );

#undef __FUNC__
#define __FUNC__ "jx_MatGenFDCreate"
void jx_MatGenFD_Create( jx_MatGenFD *mg )
{
    JX_START_FUNC_DH
    struct _jx_matgenfd *tmp =(struct _jx_matgenfd *)JX_MALLOC_DH(sizeof(struct _jx_matgenfd)); JX_CHECK_V_ERROR;
    
   *mg = tmp;
    tmp->debug = jx_Parser_dhHasSwitch(jx_parser_dh, "-debug_matgen");
    tmp->m = 9;
    tmp->px = tmp->py = 1;
    tmp->pz = 0;
    jx_Parser_dhReadInt(jx_parser_dh,"-m",&tmp->m);
    jx_Parser_dhReadInt(jx_parser_dh,"-px",&tmp->px);
    jx_Parser_dhReadInt(jx_parser_dh,"-py",&tmp->py);
    jx_Parser_dhReadInt(jx_parser_dh,"-pz",&tmp->pz);
    if (tmp->px < 1) tmp->px = 1;
    if (tmp->py < 1) tmp->py = 1;
    if (tmp->pz < 0) tmp->pz = 0;
    tmp->threeD = jx_false;
    if (tmp->pz)
    {
        tmp->threeD = jx_true;
    }
    else
    {
        tmp->pz = 1;
    }
    if (jx_Parser_dhHasSwitch(jx_parser_dh,"-threeD")) tmp->threeD = jx_true;
    tmp->a = tmp->b = tmp->c = 1.0;
    tmp->d = tmp->e = tmp->f = 0.0;
    tmp->g = tmp->h = 0.0;
    jx_Parser_dhReadDouble(jx_parser_dh,"-dx",&tmp->a);
    jx_Parser_dhReadDouble(jx_parser_dh,"-dy",&tmp->b);
    jx_Parser_dhReadDouble(jx_parser_dh,"-dz",&tmp->c);
    jx_Parser_dhReadDouble(jx_parser_dh,"-cx",&tmp->d);
    jx_Parser_dhReadDouble(jx_parser_dh,"-cy",&tmp->e);
    jx_Parser_dhReadDouble(jx_parser_dh,"-cz",&tmp->f);
    tmp->a = -1 * fabs(tmp->a);
    tmp->b = -1 * fabs(tmp->b);
    tmp->c = -1 * fabs(tmp->c);
    tmp->allocateMem = jx_true;
    tmp->A = tmp->B = tmp->C = tmp->D = tmp->E = tmp->F = tmp->G = tmp->H = jx_konstant;
    tmp->bcX1 = tmp->bcX2 = tmp->bcY1 = tmp->bcY2 = tmp->bcZ1 = tmp->bcZ2 = 0.0;
    jx_Parser_dhReadDouble(jx_parser_dh,"-bcx1",&tmp->bcX1);
    jx_Parser_dhReadDouble(jx_parser_dh,"-bcx2",&tmp->bcX2);
    jx_Parser_dhReadDouble(jx_parser_dh,"-bcy1",&tmp->bcY1);
    jx_Parser_dhReadDouble(jx_parser_dh,"-bcy2",&tmp->bcY2);
    jx_Parser_dhReadDouble(jx_parser_dh,"-bcz1",&tmp->bcZ1);
    jx_Parser_dhReadDouble(jx_parser_dh,"-bcz2",&tmp->bcZ2);
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_MatGenFD_Destroy"
void jx_MatGenFD_Destroy( jx_MatGenFD mg )
{
    JX_START_FUNC_DH
    JX_FREE_DH(mg); JX_CHECK_V_ERROR;
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_MatGenFD_Run"
void jx_MatGenFD_Run( jx_MatGenFD mg, JX_Int id, JX_Int np, jx_Mat_dh *AOut, jx_Vec_dh *rhsOut )
{
    JX_START_FUNC_DH
    jx_Mat_dh A;
    jx_Vec_dh rhs;
    jx_bool threeD = mg->threeD;
    JX_Int nnz;
    JX_Int m = mg->m;
    jx_bool debug = jx_false, striped;
    
    if (mg->debug && jx_logFile != NULL) debug = jx_true;
    striped = jx_Parser_dhHasSwitch(jx_parser_dh,"-striped");
    jx_Mat_dhCreate(AOut); JX_CHECK_V_ERROR;
    jx_Vec_dhCreate(rhsOut); JX_CHECK_V_ERROR;
    A = *AOut;
    rhs = *rhsOut;
    if (!jx_Parser_dhHasSwitch(jx_parser_dh, "-noChecks"))
    {
        if (!striped)
        {
            JX_Int npTest = mg->px * mg->py;
            
            if (threeD) npTest *= mg->pz;
            if (npTest != np)
            {
                jx_sprintf(jx_msgBuf_dh, "numbers don't match: jx_np_dh = %i, px*py*pz = %i", np, npTest);
                JX_SET_V_ERROR(jx_msgBuf_dh);
            }
        }
    }
    mg->cc = m;
    if (threeD)
    {
        m = mg->m = m * m * m;
    }
    else
    {
        m = mg->m = m * m;
    }
    mg->first = id * m;
    mg->hh = 1.0 / (mg->px * mg->cc - 1);
    if (debug)
    {
        jx_sprintf(jx_msgBuf_dh, "cc (local grid dimension) = %i", mg->cc);
        JX_SET_INFO(jx_msgBuf_dh);
        if (threeD)
        {
            jx_sprintf(jx_msgBuf_dh, "threeD = jx_true");
        }
        else
        {
            jx_sprintf(jx_msgBuf_dh, "threeD = jx_false");
        }
        JX_SET_INFO(jx_msgBuf_dh);
        jx_sprintf(jx_msgBuf_dh, "np= %i  id= %i", np, id);
        JX_SET_INFO(jx_msgBuf_dh);
    }
    mg->id = id;
    mg->np = np;
    nnz = threeD ? m*7 : m*5;
    if (mg->allocateMem)
    {
        A->rp = (JX_Int *)JX_MALLOC_DH((m+1)*sizeof(JX_Int)); JX_CHECK_V_ERROR;
        A->rp[0] = 0;
        A->cval = (JX_Int *)JX_MALLOC_DH(nnz*sizeof(JX_Int)); JX_CHECK_V_ERROR
        A->aval = (JX_Real *)JX_MALLOC_DH(nnz*sizeof(JX_Real)); JX_CHECK_V_ERROR;
    }
    rhs->n = m;
    A->m = m;
    A->n = m * mg->np;
    A->beg_row = mg->first;
    jx_isThreeD = threeD;
    if (jx_Parser_dhHasSwitch(jx_parser_dh,"-striped"))
    {
        jx_generateStriped(mg, A->rp, A->cval, A->aval, A, rhs); JX_CHECK_V_ERROR;
    }
    else
    {
        jx_generateBlocked(mg, A->rp, A->cval, A->aval, A, rhs); JX_CHECK_V_ERROR;
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_generateStriped"
void jx_generateStriped( jx_MatGenFD mg, JX_Int *rp, JX_Int *cval, JX_Real *aval, jx_Mat_dh A, jx_Vec_dh b )
{
    JX_START_FUNC_DH
    JX_Int mGlobal;
    JX_Int m = mg->m;
    JX_Int beg_row, end_row;
    JX_Int i, j, k, row;
    jx_bool threeD = mg->threeD;
    JX_Int idx = 0;
    JX_Real *stencil = mg->stencil;
    jx_bool debug = jx_false;
    JX_Int plane, nodeRemainder;
    JX_Int naborx1=0, naborx2=0, nabory1=0, nabory2=0;
    JX_Real *rhs;
    jx_bool applyBdry = jx_true;
    JX_Real hhalf;
    JX_Real bcx1 = mg->bcX1;
    JX_Real bcx2 = mg->bcX2;
    JX_Real bcy1 = mg->bcY1;
    JX_Real bcy2 = mg->bcY2;
    JX_Int nx, ny;
    
    jx_printf_dh("@@@ using striped partitioning\n");
    if (mg->debug && jx_logFile != NULL) debug = jx_true;
    m = 9;
    jx_Parser_dhReadInt(jx_parser_dh,"-m", &m);
    mGlobal = m * m;
    if (threeD) mGlobal *= m;
    i = mGlobal / mg->np;
    beg_row = i * mg->id;
    end_row = beg_row + i;
    if (mg->id == mg->np-1) end_row = mGlobal;
    nx = ny = m;
    mg->hh = 1.0 / (m - 1);
    hhalf = 0.5 * mg->hh;
    A->n = m * m;
    A->m = end_row - beg_row;
    A->beg_row = beg_row;
    jx_Vec_dhInit(b, A->m); JX_CHECK_V_ERROR;
    rhs = b->vals;
    plane = m * m;
    if (debug)
    {
        jx_fprintf(jx_logFile, "jx_generateStriped: beg_row= %i; end_row= %i; m= %i\n", beg_row+1, end_row+1, m);
    }
    for (row = beg_row; row < end_row; ++ row)
    {
        JX_Int localRow = row - beg_row;
        
        k = (row / plane);
        nodeRemainder = row - (k * plane);
        j = nodeRemainder / m;
        i = nodeRemainder % m;
        if (debug)
        {
            jx_fprintf(jx_logFile, "row= %i  x= %i  y= %i  z= %i\n", row+1, i,j,k);
        }
        jx_getstencil(mg,i,j,k);
        if (threeD)
        {
            if (k > 0)
            {
                cval[idx] = row - plane;
                aval[idx++] = JX_BACK(stencil);
            }
        }
        if (j > 0)
        {
            nabory1 = cval[idx] = row - m;
            aval[idx++] = JX_SOUTH(stencil);
        }
        if (i > 0)
        {
            naborx1 = cval[idx] = row - 1;
            aval[idx++] = JX_WEST(stencil);
        }
        cval[idx] = row;
        aval[idx++] = JX_CENTER(stencil);
        if (i < m-1)
        {
            naborx2 = cval[idx] = row + 1;
            aval[idx++] = JX_EAST(stencil);
        }
        if (j < m-1)
        {
            nabory2 = cval[idx] = row + m;
            aval[idx++] = JX_NORTH(stencil);
        }
        if (threeD)
        {
            if (k < m-1)
            {
                cval[idx] = row + plane;
                aval[idx++] = JX_FRONT(stencil);
            }
        }
        rhs[localRow] = 0.0;
        ++ localRow;
        rp[localRow] = idx;
        if (!threeD && applyBdry)
        {
            JX_Int offset = rp[localRow-1];
            JX_Int len = rp[localRow] - rp[localRow-1];
            JX_Real ctr, coeff;
            
            if (i == 0)
            {
                coeff = mg->A(mg->a, i+hhalf, j, k);
                ctr = mg->A(mg->a, i-hhalf, j, k);
                jx_setBoundary_private(row, cval+offset, aval+offset,
                            len, &(rhs[localRow-1]), bcx1, coeff, ctr, naborx2);
            }
            else if (i == nx-1)
            {
                coeff = mg->A(mg->a, i-hhalf, j, k);
                ctr = mg->A(mg->a, i+hhalf, j, k);
                jx_setBoundary_private(row, cval+offset, aval+offset,
                            len, &(rhs[localRow-1]), bcx2, coeff, ctr, naborx1);
            }
            else if (j == 0)
            {
                coeff = mg->B(mg->b, i, j+hhalf, k);
                ctr = mg->B(mg->b, i, j-hhalf, k);
                jx_setBoundary_private(row, cval+offset, aval+offset,
                            len, &(rhs[localRow-1]), bcy1, coeff, ctr, nabory2);
            }
            else if (j == ny-1)
            {
                coeff = mg->B(mg->b, i, j-hhalf, k);
                ctr = mg->B(mg->b, i, j+hhalf, k);
                jx_setBoundary_private(row, cval+offset, aval+offset,
                            len, &(rhs[localRow-1]), bcy2, coeff, ctr, nabory1);
            }
        }
    }
    JX_END_FUNC_DH
}

JX_Int jx_rownum( const jx_bool threeD,
            const JX_Int x,
            const JX_Int y,
            const JX_Int z,
            const JX_Int nx,
            const JX_Int ny,
            const JX_Int nz,
            JX_Int P,
            JX_Int Q )
{
    JX_Int p, q, r;
    JX_Int lowerx, lowery, lowerz;
    JX_Int id, startrow;
    
    p = x / nx;
    q = y / ny;
    r = z / nz;
    if (threeD)
    {
        id = r * P * Q + q * P + p;
    }
    else
    {
        id = q * P + p;
    }
    startrow = id * (nx * ny * nz);
    lowerx = nx * p;
    lowery = ny * q;
    lowerz = nz * r;
    if (threeD)
    {
        return (startrow + nx * ny * (z - lowerz) + nx * (y - lowery) + (x - lowerx));
    }
    else
    {
        return (startrow + nx * (y - lowery) + (x - lowerx));
    }
}

void jx_getstencil( jx_MatGenFD g, JX_Int ix, JX_Int iy, JX_Int iz )
{
    JX_Int k;
    JX_Real h = g->hh;
    JX_Real hhalf = h * 0.5;
    JX_Real x = h * ix;
    JX_Real y = h * iy;
    JX_Real z = h * iz;
    JX_Real cntr = 0.0;
    JX_Real *stencil = g->stencil;
    JX_Real coeff;
    jx_bool threeD = g->threeD;
    
    for (k = 0; k < 8; ++ k) stencil[k] = 0.0;
    coeff = g->A(g->a, x+hhalf, y, z);
    JX_EAST(stencil) += coeff;
    cntr += coeff;
    coeff = g->A(g->a, x-hhalf, y, z);
    JX_WEST(stencil) += coeff;
    cntr += coeff;
    coeff = g->D(g->d, x, y, z) * hhalf;
    JX_EAST(stencil) += coeff;
    JX_WEST(stencil) -= coeff;
    coeff = g->B(g->b, x, y+hhalf, z);
    JX_NORTH(stencil) += coeff;
    cntr += coeff;
    coeff = g->B(g->b, x, y-hhalf, z);
    JX_SOUTH(stencil) += coeff;
    cntr += coeff;
    coeff = g->E(g->e, x, y, z) * hhalf;
    JX_NORTH(stencil) += coeff;
    JX_SOUTH(stencil) -= coeff;
    if (threeD)
    {
        coeff = g->C(g->c, x, y, z+hhalf);
        JX_BACK(stencil) += coeff;
        cntr += coeff;
        coeff = g->C(g->c, x, y, z-hhalf);
        JX_FRONT(stencil) += coeff;
        cntr += coeff;
        coeff = g->F(g->f, x, y, z) * hhalf;
        JX_BACK(stencil) += coeff;
        JX_FRONT(stencil) -= coeff;
    }
    coeff = g->G(g->g, x, y, z);
    JX_CENTER(stencil) = h * h * coeff - cntr;
    JX_RHS(stencil) = h * h * g->H(g->h, x, y, z);
}

JX_Real jx_konstant( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z )
{
    return coeff;
}

JX_Real jx_e2_xy( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z )
{
    return exp(coeff * x * y);
}

JX_Real jx_boxThreeD( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z );

JX_Real jx_box_1( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z )
{
    static jx_bool setup = jx_false;
    JX_Real retval = coeff;
    static JX_Real dd1 = JX_BOX1_DD;
    static JX_Real dd2 = JX_BOX2_DD;
    static JX_Real dd3 = JX_BOX3_DD;
    static JX_Real ax1 = JX_BOX1_X1, ay1 = JX_BOX1_Y1;
    static JX_Real ax2 = JX_BOX1_X2, ay2 = JX_BOX1_Y2;
    static JX_Real bx1 = JX_BOX2_X1, by1 = JX_BOX2_Y1;
    static JX_Real bx2 = JX_BOX2_X2, by2 = JX_BOX2_Y2;
    static JX_Real cx1 = JX_BOX3_X1, cy1 = JX_BOX3_Y1;
    static JX_Real cx2 = JX_BOX3_X2, cy2 = JX_BOX3_Y2;
    
    if (jx_isThreeD)
    {
        return jx_boxThreeD(coeff, x, y, z);
    }
    if (!setup)
    {
        dd1 = 0.1;
        dd2 = 0.1;
        dd3 = 10;
        jx_Parser_dhReadDouble(jx_parser_dh, "-dd1", &dd1);
        jx_Parser_dhReadDouble(jx_parser_dh, "-dd2", &dd2);
        jx_Parser_dhReadDouble(jx_parser_dh, "-dd3", &dd3);
        jx_Parser_dhReadDouble(jx_parser_dh, "-box1x1", &cx1);
        jx_Parser_dhReadDouble(jx_parser_dh, "-box1x2", &cx2);
        setup = jx_true;
    }
    if (x > ax1 && x < ax2 && y > ay1 && y < ay2)
    {
        retval = dd1 * coeff;
    }
    if (x > bx1 && x < bx2 && y > by1 && y < by2)
    {
        retval = dd2 * coeff;
    }
    if (x > cx1 && x < cx2 && y > cy1 && y < cy2)
    {
        retval = dd3 * coeff;
    }
    
    return retval;
}

JX_Real jx_boxThreeD( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z )
{
    static jx_bool setup = jx_false;
    JX_Real retval = coeff;
    static JX_Real dd1 = 100;
    static JX_Real x1 = 0.2, x2 = 0.8;
    static JX_Real y1 = 0.3, y2 = 0.7;
    static JX_Real z1 = 0.4, z2 = 0.6;
    
    if (!setup)
    {
        jx_Parser_dhReadDouble(jx_parser_dh, "-dd1", &dd1);
        setup = jx_true;
    }
    if (x > x1 && x < x2 && y > y1 && y < y2 && z > z1 && z < z2)
    {
        retval = dd1 * coeff;
    }
    
    return retval;
}

JX_Real jx_box_2( JX_Real coeff, JX_Real x, JX_Real y, JX_Real z )
{
    jx_bool setup = jx_false;
    static JX_Real d1, d2;
    JX_Real retval;
    
    if (!setup)
    {
        d1 = 1;
        d2 = 2;
        jx_Parser_dhReadDouble(jx_parser_dh, "-bd1", &d1);
        jx_Parser_dhReadDouble(jx_parser_dh, "-bd2", &d2);
    }
    retval = d2;
    if (x < 0.5 && y < 0.5) retval = d1;
    if (x > 0.5 && y > 0.5) retval = d1;
    
    return (-1 * retval);
}

#undef __FUNC__
#define __FUNC__ "jx_generateBlocked"
void jx_generateBlocked( jx_MatGenFD mg, JX_Int *rp, JX_Int *cval, JX_Real *aval, jx_Mat_dh A, jx_Vec_dh b )
{
    JX_START_FUNC_DH
    jx_bool applyBdry = jx_true;
    JX_Real *stencil = mg->stencil;
    JX_Int id = mg->id;
    jx_bool threeD = mg->threeD;
    JX_Int px = mg->px, py = mg->py, pz = mg->pz;
    JX_Int p, q, r;
    JX_Int cc = mg->cc;
    JX_Int nx = cc, ny = cc, nz = cc;
    JX_Int lowerx, upperx, lowery, uppery, lowerz, upperz;
    JX_Int startRow;
    JX_Int x, y, z;
    jx_bool debug = jx_false;
    JX_Int idx = 0, localRow = 0;
    JX_Int naborx1=0, naborx2=0, nabory1=0, nabory2=0, naborz1, naborz2;
    JX_Real *rhs;
    JX_Real hhalf = 0.5 * mg->hh;
    JX_Real bcx1 = mg->bcX1;
    JX_Real bcx2 = mg->bcX2;
    JX_Real bcy1 = mg->bcY1;
    JX_Real bcy2 = mg->bcY2;
    
    jx_Vec_dhInit(b, A->m); JX_CHECK_V_ERROR;
    rhs = b->vals;
    if (mg->debug && jx_logFile != NULL) debug = jx_true;
    if (!threeD) nz = 1;
    p = id % px;
    q = ((id - p) / px) % py;
    r = (id - p - px * q) / (px * py);
    if (debug)
    {
        jx_sprintf(jx_msgBuf_dh, "this proc's position in subdomain grid: p= %i  q= %i  r= %i", p, q, r);
        JX_SET_INFO(jx_msgBuf_dh);
    }
    lowerx = nx * p;
    upperx = lowerx + nx;
    lowery = ny * q;
    uppery = lowery + ny;
    lowerz = nz * r;
    upperz = lowerz + nz;
    if (debug)
    {
        jx_sprintf(jx_msgBuf_dh, "local grid parameters: lowerx= %i  upperx= %i", lowerx, upperx);
        JX_SET_INFO(jx_msgBuf_dh);
        jx_sprintf(jx_msgBuf_dh, "local grid parameters: lowery= %i  uppery= %i", lowery, uppery);
        JX_SET_INFO(jx_msgBuf_dh);
        jx_sprintf(jx_msgBuf_dh, "local grid parameters: lowerz= %i  upperz= %i", lowerz, upperz);
        JX_SET_INFO(jx_msgBuf_dh);
    }
    startRow = mg->first;
    rp[0] = 0;
    for (z = lowerz; z < upperz; z ++)
    {
        for (y = lowery; y < uppery; y ++)
        {
            for (x = lowerx; x < upperx; x ++)
            {
                if (debug)
                {
                    jx_fprintf(jx_logFile, "row= %i  x= %i  y= %i  z= %i\n", localRow+startRow+1, x, y, z);
                }
                jx_getstencil(mg, x, y, z);
                if (threeD)
                {
                    if (z > 0)
                    {
                        naborz1 = jx_rownum(threeD, x, y, z-1, nx, ny, nz, px, py);
                        cval[idx] = naborz1;
                        aval[idx++] = JX_FRONT(stencil);
                    }
                }
                if (y > 0)
                {
                    nabory1 = jx_rownum(threeD, x, y-1, z, nx, ny, nz, px, py);
                    cval[idx] = nabory1;
                    aval[idx++] = JX_SOUTH(stencil);
                }
                if (x > 0)
                {
                    naborx1 = jx_rownum(threeD, x-1, y, z, nx, ny, nz, px, py);
                    cval[idx] = naborx1;
                    aval[idx++] = JX_WEST(stencil);
                }
                cval[idx] = localRow + startRow;
                aval[idx++] = JX_CENTER(stencil);
                if (x < nx*px-1)
                {
                    naborx2 = jx_rownum(threeD, x+1, y, z, nx, ny, nz, px, py);
                    cval[idx] = naborx2;
                    aval[idx++] = JX_EAST(stencil);
                }
                if (y < ny*py-1)
                {
                    nabory2 = jx_rownum(threeD, x, y+1, z, nx, ny, nz, px, py);
                    cval[idx] = nabory2;
                    aval[idx++] = JX_NORTH(stencil);
                }
                if (threeD)
                {
                    if (z < nz*pz-1)
                    {
                        naborz2 = jx_rownum(threeD, x, y, z+1, nx, ny, nz, px, py);
                        cval[idx] = naborz2;
                        aval[idx++] = JX_BACK(stencil);
                    }
                }
                rhs[localRow] = 0.0;
                ++ localRow;
                rp[localRow] = idx;
                if (!threeD && applyBdry)
                {
                    JX_Int globalRow = localRow + startRow - 1;
                    JX_Int offset = rp[localRow-1];
                    JX_Int len = rp[localRow] - rp[localRow-1];
                    JX_Real ctr, coeff;
                    
                    if (x == 0)
                    {
                        coeff = mg->A(mg->a, x+hhalf, y, z);
                        ctr = mg->A(mg->a, x-hhalf, y, z);
                        jx_setBoundary_private(globalRow, cval+offset, aval+offset,
                                    len, &(rhs[localRow-1]), bcx1, coeff, ctr, naborx2);
                    }
                    else if (x == nx*px-1)
                    {
                        coeff = mg->A(mg->a, x-hhalf, y, z);
                        ctr = mg->A(mg->a, x+hhalf, y, z);
                        jx_setBoundary_private(globalRow, cval+offset, aval+offset,
                                    len, &(rhs[localRow-1]), bcx2, coeff, ctr, naborx1);
                    }
                    else if (y == 0)
                    {
                        coeff = mg->B(mg->b, x, y+hhalf, z);
                        ctr = mg->B(mg->b, x, y-hhalf, z);
                        jx_setBoundary_private(globalRow, cval+offset, aval+offset,
                                    len, &(rhs[localRow-1]), bcy1, coeff, ctr, nabory2);
                    }
                    else if (y == ny*py-1)
                    {
                        coeff = mg->B(mg->b, x, y-hhalf, z);
                        ctr = mg->B(mg->b, x, y+hhalf, z);
                        jx_setBoundary_private(globalRow, cval+offset, aval+offset,
                                    len, &(rhs[localRow-1]), bcy2, coeff, ctr, nabory1);
                    }
                    else if (threeD)
                    {
                        if (z == 0)
                        {
                            coeff = mg->B(mg->b, x, y, z+hhalf);
                            ctr = mg->B(mg->b, x, y, z-hhalf);
                            jx_setBoundary_private(globalRow, cval+offset, aval+offset,
                                        len, &(rhs[localRow-1]), bcy1, coeff, ctr, naborz2);
                        }
                        else if (z == nz*nx-1)
                        {
                            coeff = mg->B(mg->b, x, y, z-hhalf);
                            ctr = mg->B(mg->b, x, y, z+hhalf);
                            jx_setBoundary_private(globalRow, cval+offset, aval+offset,
                                        len, &(rhs[localRow-1]), bcy1, coeff, ctr, naborz1);
                        }
                    }
                }
            }
        }
    }
    JX_END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "jx_setBoundary_private"
void jx_setBoundary_private( JX_Int node,
                          JX_Int *cval,
                          JX_Real *aval,
                          JX_Int len,
                          JX_Real *rhs,
                          JX_Real bc,
                          JX_Real coeff,
                          JX_Real ctr,
                          JX_Int nabor )
{
    JX_START_FUNC_DH
    JX_Int i;
    
    if (bc >= 0)
    {
       *rhs = bc;
        for (i = 0; i < len; ++ i)
        {
            if (cval[i] == node)
            {
                aval[i] = 1.0;
            }
            else
            {
                aval[i] = 0;
            }
        }
    }
    else
    {
        for (i = 0; i < len; ++ i)
        {
            if (cval[i] == node)
            {
                aval[i] += (ctr - coeff);
            }
            else if (cval[i] == nabor)
            {
                aval[i] = 2.0 * coeff;
            }
        }
    }
    JX_END_FUNC_DH
}
