/*
Adapted from:
   csr_graph.h

   Implements a CSR Graph. Part of the GGC source code. 
   Interface derived from LonestarGPU.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu> 
*/

#include "csr.h"

#include <fstream>
#include <sys/stat.h>
#include <sys/mman.h>

CSRGraph::CSRGraph() {
  init();
}

unsigned CSRGraph::init() {
  row_start = edge_dst = NULL;
  edge_data = NULL;
  node_data = NULL;
  nnodes = nedges = 0;
  device_graph = false;

  return 0;
}

unsigned CSRGraph::read(char file[]) {
  return readFromGR(file);
}


unsigned CSRGraph::readFromGR(char file[]) {
  std::ifstream cfile;
  cfile.open(file);

  // copied from GaloisCpp/trunk/src/FileGraph.h
  int masterFD = open(file, O_RDONLY);
  if (masterFD == -1) {
    printf("FileGraph::structureFromFile: unable to open %s.\n", file);
    return 1;
  }

  struct stat buf;
  int f = fstat(masterFD, &buf);
  if (f == -1) {
    printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
    abort();
  }
  size_t masterLength = buf.st_size;

  int _MAP_BASE = MAP_PRIVATE;
  //#ifdef MAP_POPULATE
  //  _MAP_BASE  |= MAP_POPULATE;
  //#endif

  void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  if (m == MAP_FAILED) {
    m = 0;
    printf("FileGraph::structureFromFile: mmap failed.\n");
    abort();
  }

//   ggc::Timer t("graphreader");
//   t.start();

  //parse file
  uint64_t* fptr = (uint64_t*)m;
  __attribute__((unused)) uint64_t version = le64toh(*fptr++);
  assert(version == 1);
  uint64_t sizeEdgeTy = le64toh(*fptr++);
  uint64_t numNodes = le64toh(*fptr++);
  uint64_t numEdges = le64toh(*fptr++);
  uint64_t *outIdx = fptr;
  fptr += numNodes;
  uint32_t *fptr32 = (uint32_t*)fptr;
  uint32_t *outs = fptr32; 
  fptr32 += numEdges;
  if (numEdges % 2) fptr32 += 1;
  unsigned  *edgeData = (unsigned *)fptr32;
	
  // cuda.
  nnodes = numNodes;
  nedges = numEdges;

  printf("nnodes=%d, nedges=%d.\n", nnodes, nedges);
  allocOnHost();

  row_start[0] = 0;

  for (unsigned ii = 0; ii < nnodes; ++ii) {
    row_start[ii+1] = le64toh(outIdx[ii]);
    //   //noutgoing[ii] = le64toh(outIdx[ii]) - le64toh(outIdx[ii - 1]);
    index_type degree = row_start[ii+1] - row_start[ii];

    for (unsigned jj = 0; jj < degree; ++jj) {
      unsigned edgeindex = row_start[ii] + jj;

      unsigned dst = le32toh(outs[edgeindex]);
      if (dst >= nnodes) printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj, edgeindex);

      edge_dst[edgeindex] = dst;

      if(sizeEdgeTy)
	edge_data[edgeindex] = edgeData[edgeindex];
    }

    progressPrint(nnodes, ii);
  }

  cfile.close();	// probably galois doesn't close its file due to mmap.
//   t.stop();

  // TODO: fix MB/s
//   printf("read %lld bytes in %d ms (%0.2f MB/s)\n\r\n", masterLength, t.duration_ms(), (masterLength / 1000.0) / (t.duration_ms()));

  return 0;
}

void CSRGraph::writeToCSR(char file[]) {
  FILE* f = fopen(file,"wb");
  if (f == NULL) {
    printf("couldn't open file %s\n",file);
    return;
  }
  // Write metadata
  fwrite(&this->nnodes, sizeof(index_type), 1, f);
  fwrite(&this->nnodes, sizeof(index_type), 1, f);
  fwrite(&this->nedges, sizeof(index_type), 1, f);

  // Write data

    fwrite(this->row_start, sizeof(index_type),
            this->nnodes + 1, f);
    fwrite(this->edge_dst, sizeof(index_type),
            this->nedges, f);
    fwrite(this->edge_data, sizeof(edge_data_type),
            this->nedges, f);

  fclose(f);

}

void CSRGraph::progressPrint(unsigned maxii, unsigned ii) {
  const unsigned nsteps = 10;
  unsigned ineachstep = (maxii / nsteps);
  if(ineachstep == 0) ineachstep = 1;
  /*if (ii == maxii) {
    printf("\t100%%\n");
    } else*/ if (ii % ineachstep == 0) {
    int progress = ((size_t) ii * 100) / maxii + 1;

    printf("\t%3d%%\r", progress);
    fflush(stdout);
  }
}

unsigned CSRGraph::allocOnHost() {
  assert(nnodes > 0);
  assert(!device_graph);

  if(row_start != NULL) // already allocated
    return true;

  row_start = (index_type *) calloc(nnodes+1, sizeof(index_type));
  edge_dst  = (index_type *) calloc(nedges, sizeof(index_type));
  edge_data = (edge_data_type *) calloc(nedges, sizeof(edge_data_type));
  node_data = (node_data_type *) calloc(nnodes, sizeof(node_data_type));

  size_t mem_usage = ((nnodes + 1) + nedges) * sizeof(index_type) 
    + (nedges) * sizeof(edge_data_type) + (nnodes) * sizeof(node_data_type);
    
  printf("Host memory for graph: %3u MB\n", mem_usage / 1048756);

  return (edge_data && row_start && edge_dst && node_data);
}


unsigned CSRGraph::allocOnDevice() {
  if(edge_dst != NULL)  // already allocated
    return true;  

  assert(edge_dst == NULL); // make sure not already allocated

  check_cuda(cudaMalloc((void **) &edge_dst, nedges * sizeof(index_type)));
  check_cuda(cudaMalloc((void **) &row_start, (nnodes+1) * sizeof(index_type)));

  check_cuda(cudaMalloc((void **) &edge_data, nedges * sizeof(edge_data_type)));
  check_cuda(cudaMalloc((void **) &node_data, nnodes * sizeof(node_data_type)));

  device_graph = true;

  return (edge_dst && edge_data && row_start && node_data);
}

void CSRGraph::copy_to_gpu(struct CSRGraph &copygraph) {
  copygraph.nnodes = nnodes;
  copygraph.nedges = nedges;
  
  assert(copygraph.allocOnDevice());

  check_cuda(cudaMemcpy(copygraph.edge_dst, edge_dst, nedges * sizeof(index_type), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(copygraph.edge_data, edge_data, nedges * sizeof(edge_data_type), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(copygraph.node_data, node_data, nnodes * sizeof(edge_data_type), cudaMemcpyHostToDevice));

  check_cuda(cudaMemcpy(copygraph.row_start, row_start, (nnodes+1) * sizeof(index_type), cudaMemcpyHostToDevice));
}


void init_trivial_graph(CSRGraph& g) {
    g.nnodes = 6;
    g.nedges = 16;
    g.row_start = (index_type *)malloc((g.nnodes+1)*sizeof(index_type));
    g.edge_dst = (index_type *)malloc(g.nedges*sizeof(edge_data_type));
    g.edge_data = (edge_data_type *)malloc(g.nedges*sizeof(edge_data_type));
    g.node_data = (node_data_type *)malloc(g.nnodes*sizeof(edge_data_type));

    g.row_start[0] = 0;
    g.row_start[1] = 2;
    g.row_start[2] = 5;
    g.row_start[3] = 9;
    g.row_start[4] = 11;
    g.row_start[5] = 13;
    g.row_start[6] = 16;

    g.edge_dst[0] = 1;
    g.edge_dst[1] = 2;

    g.edge_dst[2] = 0;
    g.edge_dst[3] = 2;
    g.edge_dst[4] = 3;

    g.edge_dst[5] = 0;
    g.edge_dst[6] = 1;
    g.edge_dst[7] = 4;
    g.edge_dst[8] = 5;

    g.edge_dst[9] = 1;
    g.edge_dst[10] = 5;

    g.edge_dst[11] = 2;
    g.edge_dst[12] = 5;

    g.edge_dst[13] = 2;
    g.edge_dst[14] = 3;
    g.edge_dst[15] = 4;

    g.edge_data[0] = 2;
    g.edge_data[1] = 4;

    g.edge_data[2] = 2;
    g.edge_data[3] = 1;
    g.edge_data[4] = 6;

    g.edge_data[5] = 4;
    g.edge_data[6] = 1;
    g.edge_data[7] = 2;
    g.edge_data[8] = 3;

    g.edge_data[9] = 6;
    g.edge_data[10] = 5;

    g.edge_data[11] = 2;
    g.edge_data[12] = 4;

    g.edge_data[13] = 3;
    g.edge_data[14] = 5;
    g.edge_data[15] = 4;
}

