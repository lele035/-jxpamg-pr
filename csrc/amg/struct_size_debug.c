#include "jx_pamg.h"
#include <stdio.h>
static int _dummy_JXPAMG_PR_ResidualProbe_debug(void) {
  jx_CSRMatrix *tmp=NULL; jx_ParCSRMatrix *ptmp=NULL;
  printf("LIB_UNIT: ParCSR=%zu CSR=%zu JX_Int=%zu JX_Real=%zu MPI_Comm=%zu\n",
    sizeof(*ptmp), sizeof(*tmp), sizeof(JX_Int), sizeof(JX_Real), sizeof(MPI_Comm));
  return 0;
}
