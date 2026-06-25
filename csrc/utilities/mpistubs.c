//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

#include "jx_util.h"

JX_Int
jx_MPI_Init( int *argc, char ***argv )
{
   return (JX_Int) MPI_Init(argc, argv);
}

JX_Int
jx_MPI_Finalize()
{
   return (JX_Int) MPI_Finalize();
}

JX_Int
jx_MPI_Abort( MPI_Comm comm, JX_Int errorcode )
{
   return (JX_Int) MPI_Abort(comm, (int)errorcode);
}

JX_Real
jx_MPI_Wtime()
{
   return MPI_Wtime();
}

JX_Real
jx_MPI_Wtick()
{
   return MPI_Wtick();
}

JX_Int
jx_MPI_Barrier( MPI_Comm comm )
{
   return (JX_Int) MPI_Barrier(comm);
}

JX_Int
jx_MPI_Comm_create( MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm )
{
   return (JX_Int) MPI_Comm_create(comm, group, newcomm);
}

JX_Int
jx_MPI_Comm_dup( MPI_Comm comm, MPI_Comm *newcomm )
{
   return (JX_Int) MPI_Comm_dup(comm, newcomm);
}

JX_Int
jx_MPI_Comm_size( MPI_Comm comm, JX_Int *size )
{
   int mpi_size;
   JX_Int ierr;
   ierr = (JX_Int) MPI_Comm_size(comm, &mpi_size);
   *size = (JX_Int) mpi_size;
   return ierr;
}

JX_Int
jx_MPI_Comm_rank( MPI_Comm comm, JX_Int *rank )
{ 
   int mpi_rank;
   JX_Int ierr;
   ierr = (JX_Int) MPI_Comm_rank(comm, &mpi_rank);
   *rank = (JX_Int) mpi_rank;
   return ierr;
}

JX_Int
jx_MPI_Comm_free( MPI_Comm *comm )
{
   return (JX_Int) MPI_Comm_free(comm);
}

JX_Int
jx_MPI_Comm_group( MPI_Comm comm, MPI_Group *group )
{
   return (JX_Int) MPI_Comm_group(comm, group);
}

JX_Int
jx_MPI_Comm_split( MPI_Comm comm, JX_Int n, JX_Int m, MPI_Comm *comms )
{
   return (JX_Int) MPI_Comm_split(comm, (int)n, (int)m, comms);
}

JX_Int
jx_MPI_Group_incl( MPI_Group group, JX_Int n, JX_Int *ranks, MPI_Group *newgroup )
{
   int *mpi_ranks;
   JX_Int  i;
   JX_Int  ierr;

   mpi_ranks = jx_TAlloc(int, n);
   for (i = 0; i < n; i++)
   {
      mpi_ranks[i] = (int) ranks[i];
   }
   ierr = (JX_Int) MPI_Group_incl(group, (int)n, mpi_ranks, newgroup);
   jx_TFree(mpi_ranks);

   return ierr;
}

JX_Int
jx_MPI_Group_free( MPI_Group *group )
{
   return (JX_Int) MPI_Group_free(group);
}

JX_Int
jx_MPI_Address( void *location, MPI_Aint *address )
{
   return (JX_Int) MPI_Get_address(location, address);
}

JX_Int
jx_MPI_Get_count( MPI_Status *status, MPI_Datatype datatype, JX_Int *count )
{
   int mpi_count;
   JX_Int ierr;
   ierr = (JX_Int) MPI_Get_count(status, datatype, &mpi_count);
   *count = (JX_Int) mpi_count;
   return ierr;
}

JX_Int
jx_MPI_Alltoall( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, JX_Int recvcount, MPI_Datatype recvtype, MPI_Comm comm )
{
   return (JX_Int) MPI_Alltoall(sendbuf, (int)sendcount, sendtype, recvbuf, (int)recvcount, recvtype, comm);
}

JX_Int
jx_MPI_Allgather( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype, void *recvbuf, JX_Int recvcount, MPI_Datatype recvtype, MPI_Comm comm ) 
{
   return (JX_Int) MPI_Allgather(sendbuf, (int)sendcount, sendtype, recvbuf, (int)recvcount, recvtype, comm);
}

JX_Int
jx_MPI_Allgatherv( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, JX_Int *recvcounts, JX_Int *displs, MPI_Datatype recvtype, MPI_Comm comm ) 
{
   int *mpi_recvcounts, *mpi_displs, csize;
   JX_Int  i;
   JX_Int  ierr;

   MPI_Comm_size(comm, &csize);
   mpi_recvcounts = jx_TAlloc(int, csize);
   mpi_displs = jx_TAlloc(int, csize);
   for (i = 0; i < csize; i++)
   {
      mpi_recvcounts[i] = (int) recvcounts[i];
      mpi_displs[i] = (int) displs[i];
   }
   ierr = (JX_Int) MPI_Allgatherv(sendbuf, (int)sendcount, sendtype, recvbuf, mpi_recvcounts, mpi_displs, recvtype, comm);
   jx_TFree(mpi_recvcounts);
   jx_TFree(mpi_displs);

   return ierr;
}

JX_Int
jx_MPI_Gather( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  JX_Int recvcount, MPI_Datatype recvtype, JX_Int root, MPI_Comm comm )
{
   return (JX_Int) MPI_Gather(sendbuf, (int) sendcount, sendtype, recvbuf, (int) recvcount, recvtype, (int)root, comm);
}

JX_Int
jx_MPI_Gatherv( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  JX_Int *recvcounts, JX_Int *displs, MPI_Datatype recvtype, JX_Int root, MPI_Comm comm )
{
   int *mpi_recvcounts = NULL;
   int *mpi_displs = NULL;
   int csize, croot;
   JX_Int  i;
   JX_Int  ierr;

   MPI_Comm_size(comm, &csize);
   MPI_Comm_rank(comm, &croot);
   if (croot == (int) root)
   {
      mpi_recvcounts = jx_TAlloc(int, csize);
      mpi_displs = jx_TAlloc(int, csize);
      for (i = 0; i < csize; i++)
      {
         mpi_recvcounts[i] = (int) recvcounts[i];
         mpi_displs[i] = (int) displs[i];
      }
   }
   ierr = (JX_Int) MPI_Gatherv(sendbuf, (int)sendcount, sendtype, recvbuf, mpi_recvcounts, mpi_displs, recvtype, (int) root, comm);
   jx_TFree(mpi_recvcounts);
   jx_TFree(mpi_displs);

   return ierr;
}

JX_Int
jx_MPI_Scatter( void *sendbuf, JX_Int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, JX_Int recvcount, MPI_Datatype recvtype, JX_Int root, MPI_Comm comm )
{
   return (JX_Int) MPI_Scatter(sendbuf, (int)sendcount, sendtype, recvbuf, (int)recvcount, recvtype, (int)root, comm);
}

JX_Int
jx_MPI_Scatterv( void *sendbuf, JX_Int *sendcounts, JX_Int *displs, MPI_Datatype sendtype,
                   void *recvbuf, JX_Int recvcount, MPI_Datatype recvtype, JX_Int root, MPI_Comm comm )
{
   int *mpi_sendcounts = NULL;
   int *mpi_displs = NULL;
   int csize, croot;
   JX_Int  i;
   JX_Int  ierr;

   MPI_Comm_size(comm, &csize);
   MPI_Comm_rank(comm, &croot);
   if (croot == (int) root)
   {
      mpi_sendcounts = jx_TAlloc(int, csize);
      mpi_displs = jx_TAlloc(int, csize);
      for (i = 0; i < csize; i++)
      {
         mpi_sendcounts[i] = (int) sendcounts[i];
         mpi_displs[i] = (int) displs[i];
      }
   }
   ierr = (JX_Int) MPI_Scatterv(sendbuf, mpi_sendcounts, mpi_displs, sendtype, recvbuf, (int) recvcount, recvtype, (int) root, comm);
   jx_TFree(mpi_sendcounts);
   jx_TFree(mpi_displs);

   return ierr;
}

JX_Int
jx_MPI_Bcast( void *buffer, JX_Int count, MPI_Datatype datatype, JX_Int root, MPI_Comm comm ) 
{ 
   return (JX_Int) MPI_Bcast(buffer, (int)count, datatype, (int)root, comm);
}

JX_Int
jx_MPI_Send( void *buf, JX_Int count, MPI_Datatype datatype, JX_Int dest, JX_Int tag, MPI_Comm comm ) 
{ 
   return (JX_Int) MPI_Send(buf, (int)count, datatype, (int)dest, (int)tag, comm);
}

JX_Int
jx_MPI_Recv( void *buf, JX_Int count, MPI_Datatype  datatype, JX_Int source, JX_Int tag, MPI_Comm comm, MPI_Status *status )
{ 
   return (JX_Int) MPI_Recv(buf, (int)count, datatype, (int)source, (int)tag, comm, status);
}

JX_Int
jx_MPI_Isend( void *buf, JX_Int count, MPI_Datatype datatype, JX_Int dest, JX_Int tag, MPI_Comm comm, MPI_Request *request )
{ 
   return (JX_Int) MPI_Isend(buf, (int)count, datatype, (int)dest, (int)tag, comm, request);
}

JX_Int
jx_MPI_Irecv( void *buf, JX_Int count, MPI_Datatype datatype, JX_Int source, JX_Int tag, MPI_Comm comm, MPI_Request *request )
{ 
   return (JX_Int) MPI_Irecv(buf, (int)count, datatype, (int)source, (int)tag, comm, request);
}

JX_Int
jx_MPI_Send_init( void *buf, JX_Int count, MPI_Datatype datatype, JX_Int dest, JX_Int tag, MPI_Comm comm, MPI_Request *request )
{
   return (JX_Int) MPI_Send_init(buf, (int)count, datatype, (int)dest, (int)tag, comm, request);
}

JX_Int
jx_MPI_Recv_init( void *buf, JX_Int count, MPI_Datatype datatype, JX_Int dest,
                     JX_Int tag, MPI_Comm comm, MPI_Request *request )
{
   return (JX_Int) MPI_Recv_init(buf, (int)count, datatype, (int)dest, (int)tag, comm, request);
}

JX_Int
jx_MPI_Irsend( void *buf, JX_Int count, MPI_Datatype datatype,
                  JX_Int dest, JX_Int tag, MPI_Comm comm, MPI_Request *request )
{
   return (JX_Int) MPI_Irsend(buf, (int)count, datatype, (int)dest, (int)tag, comm, request);
}

JX_Int
jx_MPI_Startall( JX_Int count, MPI_Request *array_of_requests )
{
   return (JX_Int) MPI_Startall((int)count, array_of_requests);
}

JX_Int
jx_MPI_Probe( JX_Int source, JX_Int tag, MPI_Comm comm, MPI_Status *status )
{
   return (JX_Int) MPI_Probe((int)source, (int)tag, comm, status);
}

JX_Int
jx_MPI_Iprobe( JX_Int source, JX_Int tag, MPI_Comm comm, JX_Int *flag, MPI_Status *status )
{
   int mpi_flag;
   JX_Int ierr;
   ierr = (JX_Int) MPI_Iprobe((int)source, (int)tag, comm, &mpi_flag, status);
  *flag = (JX_Int) mpi_flag;
   return ierr;
}

JX_Int
jx_MPI_Test( MPI_Request *request, JX_Int *flag, MPI_Status *status )
{
   int mpi_flag;
   JX_Int ierr;
   ierr = (JX_Int) MPI_Test(request, &mpi_flag, status);
  *flag = (JX_Int) mpi_flag;
   return ierr;
}

JX_Int
jx_MPI_Testall( JX_Int count, MPI_Request *array_of_requests, JX_Int *flag, MPI_Status *array_of_statuses )
{
   int mpi_flag;
   JX_Int ierr;
   ierr = (JX_Int) MPI_Testall((int)count, array_of_requests, &mpi_flag, array_of_statuses);
  *flag = (JX_Int) mpi_flag;
   return ierr;
}

JX_Int
jx_MPI_Wait( MPI_Request *request, MPI_Status *status )
{
   return (JX_Int) MPI_Wait(request, status);
}

JX_Int
jx_MPI_Waitall( JX_Int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses )
{
   return (JX_Int) MPI_Waitall((int)count, array_of_requests, array_of_statuses);
}

JX_Int
jx_MPI_Waitany( JX_Int count, MPI_Request *array_of_requests, JX_Int *index, MPI_Status *status )
{
   int mpi_index;
   JX_Int ierr;
   ierr = (JX_Int) MPI_Waitany((int)count, array_of_requests, &mpi_index, status);
  *index = (JX_Int) mpi_index;
   return ierr;
}

JX_Int
jx_MPI_Allreduce( void *sendbuf, void *recvbuf, JX_Int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm )
{
   return (JX_Int) MPI_Allreduce(sendbuf, recvbuf, (int)count, datatype, op, comm);
}

JX_Int
jx_MPI_Reduce( void *sendbuf, void *recvbuf, JX_Int count, MPI_Datatype datatype, MPI_Op op, JX_Int root, MPI_Comm comm )
{
   return (JX_Int) MPI_Reduce(sendbuf, recvbuf, (int)count, datatype, op, (int)root, comm);
}

JX_Int
jx_MPI_Scan( void *sendbuf, void *recvbuf, JX_Int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm )
{ 
   return (JX_Int) MPI_Scan(sendbuf, recvbuf, (int)count, datatype, op, comm);
}

JX_Int
jx_MPI_Request_free( MPI_Request *request )
{
   return (JX_Int) MPI_Request_free(request);
}

JX_Int
jx_MPI_Type_contiguous( JX_Int count, MPI_Datatype oldtype, MPI_Datatype *newtype )
{
   return (JX_Int) MPI_Type_contiguous((int)count, oldtype, newtype);
}

JX_Int
jx_MPI_Type_vector( JX_Int count, JX_Int blocklength, JX_Int stride, MPI_Datatype oldtype, MPI_Datatype *newtype )
{
   return (JX_Int) MPI_Type_vector((int)count, (int)blocklength, (int)stride, oldtype, newtype);
}

JX_Int
jx_MPI_Type_hvector( JX_Int count, JX_Int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype )
{
   return (JX_Int) MPI_Type_create_hvector((int)count, (int)blocklength, stride, oldtype, newtype);
}

JX_Int
jx_MPI_Type_struct( JX_Int count, JX_Int *array_of_blocklengths, MPI_Aint *array_of_displacements,
                       MPI_Datatype *array_of_types, MPI_Datatype *newtype )
{
   int *mpi_array_of_blocklengths;
   JX_Int  i;
   JX_Int  ierr;

   mpi_array_of_blocklengths = jx_TAlloc(int, count);
   for (i = 0; i < count; i++)
   {
      mpi_array_of_blocklengths[i] = (int) array_of_blocklengths[i];
   }
   ierr = (JX_Int) MPI_Type_create_struct((int)count, mpi_array_of_blocklengths, array_of_displacements, array_of_types, newtype);
   jx_TFree(mpi_array_of_blocklengths);

   return ierr;
}

JX_Int
jx_MPI_Type_commit( MPI_Datatype *datatype )
{
   return (JX_Int) MPI_Type_commit(datatype);
}

JX_Int
jx_MPI_Type_free( MPI_Datatype *datatype )
{
   return (JX_Int) MPI_Type_free(datatype);
}

JX_Int
jx_MPI_Op_free( MPI_Op *op )
{
   return (JX_Int) MPI_Op_free(op);
}

JX_Int
jx_MPI_Op_create( MPI_User_function *function, int commute, MPI_Op *op )
{
   return (JX_Int) MPI_Op_create(function, commute, op);
}
