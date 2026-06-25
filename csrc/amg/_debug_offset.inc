   fprintf(stderr, "LIB_OFF: diag=%zu offd=%zu data=%zu comm=%zu psz=%zu csz=%zu i=%zu r=%zu c=%zu",
     (size_t)((char*)&((jx_ParCSRMatrix*)0)->diag - (char*)0),
     (size_t)((char*)&((jx_ParCSRMatrix*)0)->offd - (char*)0),
     (size_t)((char*)&((jx_CSRMatrix*)0)->data - (char*)0),
     (size_t)((char*)&((jx_ParCSRMatrix*)0)->comm - (char*)0),
     sizeof(jx_ParCSRMatrix), sizeof(jx_CSRMatrix),
     sizeof(JX_Int), sizeof(JX_Real), sizeof(MPI_Comm));
