/**
 * @file
 * @brief This file implements the MPI-backend of ArgoDSM
 * @copyright Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.
 */
#include<cstddef>
#include<vector>

#include "env/env.hpp"
#include "signal/signal.hpp"
#include "virtual_memory/virtual_memory.hpp"
#include "data_distribution/global_ptr.hpp"
#include "swdsm.h"
#include "write_buffer.hpp"

namespace dd = argo::data_distribution;
namespace vm = argo::virtual_memory;
namespace sig = argo::signal;
namespace env = argo::env;

/** @brief For matching threads to more sensible thread IDs */
pthread_t tid[NUM_THREADS] = {0};

/*Barrier*/
/** @brief  Locks access to part that does SD in the global barrier */
pthread_mutex_t barriermutex = PTHREAD_MUTEX_INITIALIZER;
/** @brief Thread local barrier used to first wait for all local threads in the global barrier*/
pthread_barrier_t *threadbarrier;


/*Pagecache*/
/** @brief  Size of the cache in number of pages*/
unsigned long cachesize;
/** @brief  The maximum number of pages load_cache_entry will fetch remotely */
std::size_t load_size;
/** @brief  Offset off the cache in the backing file*/
unsigned long cacheoffset;
/** @brief  Keeps state, tag and dirty bit of the cache*/
control_data * cacheControl;
/** @brief  keeps track of readers and writers*/
unsigned long *globalSharers;
/** @brief  size of pyxis directory*/
unsigned long classificationSize;
/** @brief  Tracks if a page is touched this epoch*/
argo_byte * touchedcache;
/** @brief  The local page cache*/
char* cacheData;
/** @brief Copy of the local cache to keep twinpages for later being able to DIFF stores */
char * pagecopy;
/** @brief Protects the pagecache */
pthread_mutex_t cachemutex = PTHREAD_MUTEX_INITIALIZER;

/*Writebuffer*/
/** @brief A write buffer storing cache indices */
write_buffer<std::size_t>* argo_write_buffer;

/*MPI and Comm*/
/** @brief  A copy of MPI_COMM_WORLD group to split up processes into smaller groups*/
/** @todo This can be removed now when we are only running 1 process per ArgoDSM node */
MPI_Group startgroup;
/** @brief  A group of all processes that are executing the main thread */
/** @todo This can be removed now when we are only running 1 process per ArgoDSM node */
MPI_Group workgroup;
/** @brief Communicator can be replaced with MPI_COMM_WORLD*/
MPI_Comm workcomm;
/** @brief MPI window for communicating pyxis directory*/
MPI_Win sharerWindow;
/** @brief MPI window for communicating global locks*/
MPI_Win lockWindow;
/** @brief MPI windows for reading and writing data in global address space */
MPI_Win *globalDataWindow;
/** @brief MPI data structure for sending cache control data*/
MPI_Datatype mpi_control_data;
/** @brief MPI data structure for a block containing an ArgoDSM cacheline of pages */
MPI_Datatype cacheblock;
/** @brief number of MPI processes / ArgoDSM nodes */
int numtasks;
/** @brief  rank/process ID in the MPI/ArgoDSM runtime*/
int rank;
/** @brief rank/process ID in the MPI/ArgoDSM runtime*/
int workrank;
/** @brief tracking which windows are used for reading and writing global address space*/
char * barwindowsused;
/** @brief Semaphore protecting infiniband accesses*/
/** @todo replace with a (qd?)lock */
sem_t ibsem;

/*Loading and Prefetching*/
/**
 * @brief load into cache helper function
 * @param aligned_access_offset memory offset to load into the cache
 * @pre aligned_access_offset must be aligned as CACHELINE*pagesize
 */
void load_cache_entry(std::size_t aligned_access_offset);

/*Global lock*/
/** @brief  Local flags we spin on for the global lock*/
unsigned long * lockbuffer;
/** @brief  Protects the global lock so only 1 thread can have a global lock at a time */
sem_t globallocksem;
/** @brief  Keeps track of what local flag we should spin on per lock*/
int locknumber=0;

/*Global allocation*/
/** @brief  Keeps track of allocated memory in the global address space*/
unsigned long *allocationOffset;
/** @brief  Protects access to global allocator*/
pthread_mutex_t gmallocmutex = PTHREAD_MUTEX_INITIALIZER;

/*Common*/
/** @brief  Points to start of global address space*/
void * startAddr;
/** @brief  Points to start of global address space this process is serving */
char* globalData;
/** @brief  Size of global address space*/
unsigned long size_of_all;
/** @brief  Size of this process part of global address space*/
unsigned long size_of_chunk;
/** @brief  size of a page */
static const unsigned int pagesize = 4096;
/** @brief  Magic value for invalid cacheindices */
unsigned long GLOBAL_NULL;
/** @brief  Statistics */
argo_statistics stats;

/* CSPext: added this area */
/* Data Replication and Rebuild */
/** @brief  Points to start of replication area */
char* replData;
/* CSPext: Create replDataWindow */
/** @brief MPI windows for reading and writing duplicated data in global address space */
MPI_Win *replDataWindow;
/** @brief  Node alternation table */
node_alternation_table* node_alter_tbl;
/** @brief  MPI window for updating node alternation table */
MPI_Win* node_alter_tbl_window;
/** @brief  Keep track of vm::map_memory offset for dynamic allocation in data recovery */
std::size_t vm_map_offset_record;
/** @brief size of this process replication address space */
unsigned long size_of_replication;
/** @brief (local) Whether to use replication or not */
bool useReplication;
/** @brief (local) Record of pages with no repl (for EC + redirect) */
bool* no_repl_page;

/*First-Touch policy*/
/** @brief  Holds the owner and backing offset of a page */
std::uintptr_t *global_owners_dir;
/** @brief  Holds the backing offsets of the nodes */
std::uintptr_t *global_offsets_tbl;
/** @brief  Size of the owners directory */
std::size_t owners_dir_size;
/** @brief  MPI window for communicating owners directory */
MPI_Win owners_dir_window;
/** @brief  MPI window for communicating offsets table */
MPI_Win offsets_tbl_window;
/** @brief  Spinlock to avoid "spinning" on the semaphore */
std::mutex spin_mutex;

namespace {
	/** @brief constant for invalid ArgoDSM node */
	constexpr unsigned long invalid_node = static_cast<unsigned long>(-1);
}

unsigned long isPowerOf2(unsigned long x){
  unsigned long retval =  ((x & (x - 1)) == 0); //Checks if x is power of 2 (or zero)
  return retval;
}

int argo_get_local_tid(){
	int i;
	for(i = 0; i < NUM_THREADS; i++){
		if(pthread_equal(tid[i],pthread_self())){
			return i;
		}
	}
	return 0;
}

int argo_get_global_tid(){
	int i;
	for(i = 0; i < NUM_THREADS; i++){
		if(pthread_equal(tid[i],pthread_self())){
			return ((getID()*NUM_THREADS) + i);
		}
	}
	return 0;
}


void argo_register_thread(){
	int i;
	sem_wait(&ibsem);
	for(i = 0; i < NUM_THREADS; i++){
		if(tid[i] == 0){
			tid[i] = pthread_self();
			break;
		}
	}
	sem_post(&ibsem);
	pthread_barrier_wait(&threadbarrier[NUM_THREADS]);
}


void argo_pin_threads(){

  cpu_set_t cpuset;
  int s;
  argo_register_thread();
  sem_wait(&ibsem);
  CPU_ZERO(&cpuset);
  int pinto = argo_get_local_tid();
  CPU_SET(pinto, &cpuset);

  s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (s != 0){
    printf("PINNING ERROR\n");
    argo_finalize();
  }
  sem_post(&ibsem);
}


//Get cacheindex
unsigned long getCacheIndex(unsigned long addr){
	unsigned long index = (addr/pagesize) % cachesize;
	return index;
}

void init_mpi_struct(void){
	//init our struct coherence unit to work in mpi.
	const int blocklen[3] = { 1,1,1};
	MPI_Aint offsets[3];
	offsets[0] = 0;  offsets[1] = sizeof(argo_byte)*1;  offsets[2] = sizeof(argo_byte)*2;

	MPI_Datatype types[3] = {MPI_BYTE,MPI_BYTE,MPI_UNSIGNED_LONG};
	MPI_Type_create_struct(3,blocklen, offsets, types, &mpi_control_data);

	MPI_Type_commit(&mpi_control_data);
}


void init_mpi_cacheblock(void){
	//init our struct coherence unit to work in mpi.
	MPI_Type_contiguous(pagesize*CACHELINE,MPI_BYTE,&cacheblock);
	MPI_Type_commit(&cacheblock);
}

/**
 * @brief align an offset into a memory region to the beginning of its size block
 * @param offset the unaligned offset
 * @param size the size of each block
 * @return the beginning of the block of size size where offset is located
 */
inline std::size_t align_backwards(std::size_t offset, std::size_t size) {
	return (offset / size) * size;
}

void handler(int sig, siginfo_t *si, void *unused){
	UNUSED_PARAM(sig);
	UNUSED_PARAM(unused);
	double t1 = MPI_Wtime();
	unsigned long tag;
	argo_byte owner,state;
	/* compute offset in distributed memory in bytes, always positive */
	const std::size_t access_offset = static_cast<char*>(si->si_addr) - static_cast<char*>(startAddr);

	/* align access offset to cacheline */
	const std::size_t aligned_access_offset = align_backwards(access_offset, CACHELINE*pagesize);
	unsigned long classidx = get_classification_index(aligned_access_offset);

	/* compute start pointer of cacheline. char* has byte-wise arithmetics */
	char* const aligned_access_ptr = static_cast<char*>(startAddr) + aligned_access_offset;
	unsigned long startIndex = getCacheIndex(aligned_access_offset);

	/* Get homenode and offset, protect with ibsem if first touch */
	/* CSP: First touch not important for now */
	argo::node_id_t homenode;
	if(dd::is_first_touch_policy()){
		std::lock_guard<std::mutex> lock(spin_mutex);
		sem_wait(&ibsem);
		homenode = get_homenode(aligned_access_offset);
		sem_post(&ibsem);
	}else{
		homenode = get_homenode(aligned_access_offset);
	}

	unsigned long id = 1 << getID();
	unsigned long invid = ~id;

	pthread_mutex_lock(&cachemutex);

	// CSPext: For replication to occur, we cannot enter this branch
	// CSPext: If we have time, we can try to make it work with this branch
	/* page is local */
	// if(homenode == (getID())){
	// 	int n;
	// 	sem_wait(&ibsem);
	// 	unsigned long sharers;
	// 	MPI_Win_lock(MPI_LOCK_SHARED, workrank, 0, sharerWindow);
	// 	unsigned long prevsharer = (globalSharers[classidx])&id;
	// 	MPI_Win_unlock(workrank, sharerWindow);
	// 	if(prevsharer != id){
	// 		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, sharerWindow);
	// 		sharers = globalSharers[classidx];
	// 		globalSharers[classidx] |= id;
	// 		MPI_Win_unlock(workrank, sharerWindow);
	// 		if(sharers != 0 && sharers != id && isPowerOf2(sharers)){
	// 			unsigned long ownid = sharers&invid;
	// 			unsigned long owner = workrank;
	// 			for(n=0; n<numtasks; n++){
	// 				if((unsigned long)(1<<n)==ownid){
	// 					owner = n; //just get rank...
	// 					break;
	// 				}
	// 			}
	// 			if(owner==(unsigned long)workrank){
	// 				throw "bad owner in local access";
	// 			}
	// 			else{
	// 				/* update remote private holder to shared */
	// 				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner, 0, sharerWindow);
	// 				MPI_Accumulate(&id, 1, MPI_LONG, owner, classidx,1,MPI_LONG,MPI_BOR,sharerWindow);
	// 				MPI_Win_unlock(owner, sharerWindow);
	// 			}
	// 		}
	// 		/* set page to permit reads and map it to the page cache */
	// 		/** @todo Set cache offset to a variable instead of calculating it here */
	// 		vm::map_memory(aligned_access_ptr, pagesize*CACHELINE, cacheoffset+offset, PROT_READ);

	// 	}
	// 	else{

	// 		/* get current sharers/writers and then add your own id */
	// 		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, sharerWindow);
	// 		unsigned long sharers = globalSharers[classidx];
	// 		unsigned long writers = globalSharers[classidx+1];
	// 		globalSharers[classidx+1] |= id;
	// 		MPI_Win_unlock(workrank, sharerWindow);

	// 		/* remote single writer */
	// 		if(writers != id && writers != 0 && isPowerOf2(writers&invid)){
	// 			int n;
	// 			for(n=0; n<numtasks; n++){
	// 				if(((unsigned long)(1<<n))==(writers&invid)){
	// 					owner = n; //just get rank...
	// 					break;
	// 				}
	// 			}
	// 			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner, 0, sharerWindow);
	// 			MPI_Accumulate(&id, 1, MPI_LONG, owner, classidx+1,1,MPI_LONG,MPI_BOR,sharerWindow);
	// 			MPI_Win_unlock(owner, sharerWindow);
	// 		}
	// 		else if(writers == id || writers == 0){
	// 			int n;
	// 			for(n=0; n<numtasks; n++){
	// 				if(n != workrank && ((1<<n)&sharers) != 0){
	// 					MPI_Win_lock(MPI_LOCK_EXCLUSIVE, n, 0, sharerWindow);
	// 					MPI_Accumulate(&id, 1, MPI_LONG, n, classidx+1,1,MPI_LONG,MPI_BOR,sharerWindow);
	// 					MPI_Win_unlock(n, sharerWindow);
	// 				}
	// 			}
	// 		}
	// 		/* set page to permit read/write and map it to the page cache */
	// 		vm::map_memory(aligned_access_ptr, pagesize*CACHELINE, cacheoffset+offset, PROT_READ|PROT_WRITE);

	// 	}

	// 	sem_post(&ibsem);
	// 	pthread_mutex_unlock(&cachemutex);
	// 	return;
	// }

	state  = cacheControl[startIndex].state;
	tag = cacheControl[startIndex].tag;

	/* CSP: State of cachline is invalid or the cacheline has a valid page but not the one we are looking for => eviction required */
	if(state == INVALID || (tag != aligned_access_offset && tag != GLOBAL_NULL)) {
		load_cache_entry(aligned_access_offset);
		pthread_mutex_unlock(&cachemutex);
		double t2 = MPI_Wtime();
		stats.loadtime+=t2-t1;
		return;
	}

	/*CSP: These 2 lines not relevant for us */
	unsigned long line = startIndex / CACHELINE;
	line *= CACHELINE;

	if(cacheControl[line].dirty == DIRTY){
		pthread_mutex_unlock(&cachemutex);
		return;
	}


	touchedcache[line] = 1;
	cacheControl[line].dirty = DIRTY;

	sem_wait(&ibsem);
	/* CSP: Workrank = Node ID */
	MPI_Win_lock(MPI_LOCK_SHARED, workrank, 0, sharerWindow);
	/* CSP: globalShares = Pyxis directory array */
	unsigned long writers = globalSharers[classidx+1];
	unsigned long sharers = globalSharers[classidx];
	MPI_Win_unlock(workrank, sharerWindow);
	/* Either already registered write - or 1 or 0 other writers already cached */
	if(writers != id && isPowerOf2(writers)){
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, sharerWindow);
		globalSharers[classidx+1] |= id; //register locally
		MPI_Win_unlock(workrank, sharerWindow);

		/* register and get latest sharers / writers */
		MPI_Win_lock(MPI_LOCK_SHARED, homenode, 0, sharerWindow);
		MPI_Get_accumulate(&id, 1,MPI_LONG,&writers,1,MPI_LONG,homenode,
			classidx+1,1,MPI_LONG,MPI_BOR,sharerWindow);
		MPI_Get(&sharers,1, MPI_LONG, homenode, classidx, 1,MPI_LONG,sharerWindow);
		MPI_Win_unlock(homenode, sharerWindow);
		/* We get result of accumulation before operation so we need to account for that */
		writers |= id;
		/* Just add the (potentially) new sharers fetched to local copy */
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, sharerWindow);
		globalSharers[classidx] |= sharers;
		MPI_Win_unlock(workrank, sharerWindow);

		/* check if we need to update */
		/* CSP: There is a single writer and your are not a writer */
		if(writers != id && writers != 0 && isPowerOf2(writers&invid)){
			int n;
			for(n=0; n<numtasks; n++){
				if(((unsigned long)(1<<n))==(writers&invid)){
					owner = n; //just get rank...
					break;
				}
			}
			/* CSP: Update Pyxis directory of the single writer */
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner, 0, sharerWindow);
			MPI_Accumulate(&id, 1, MPI_LONG, owner, classidx+1,1,MPI_LONG,MPI_BOR,sharerWindow);
			MPI_Win_unlock(owner, sharerWindow);
		}
		/* CSP: You are the only writer or there are no writers */
		else if(writers==id || writers==0){
			int n;
			for(n=0; n<numtasks; n++){
				/* CSP: Check for sharers and update their Pyxis directory */
				if(n != workrank && ((1<<n)&sharers) != 0){
					MPI_Win_lock(MPI_LOCK_EXCLUSIVE, n, 0, sharerWindow);
					MPI_Accumulate(&id, 1, MPI_LONG, n, classidx+1,1,MPI_LONG,MPI_BOR,sharerWindow);
					MPI_Win_unlock(n, sharerWindow);
				}
			}
		}
	}
	unsigned char * copy = (unsigned char *)(pagecopy + line*pagesize);
	memcpy(copy,aligned_access_ptr,CACHELINE*pagesize);
	argo_write_buffer->add(startIndex);
	sem_post(&ibsem);
	mprotect(aligned_access_ptr, pagesize*CACHELINE,PROT_WRITE|PROT_READ);
	pthread_mutex_unlock(&cachemutex);
	double t2 = MPI_Wtime();
	stats.storetime += t2-t1;
	return;
}


argo::node_id_t get_homenode(std::size_t addr){
	dd::global_ptr<char> gptr(reinterpret_cast<char*>(
			addr + reinterpret_cast<unsigned long>(startAddr)), true, false);
	return gptr.node();
}

argo::node_id_t peek_homenode(std::size_t addr) {
	dd::global_ptr<char> gptr(reinterpret_cast<char*>(
			addr + reinterpret_cast<unsigned long>(startAddr)), false, false);
	return gptr.peek_node();
}

std::size_t get_offset(std::size_t addr){
	dd::global_ptr<char> gptr(reinterpret_cast<char*>(
			addr + reinterpret_cast<unsigned long>(startAddr)), false, true);
	return gptr.offset();
}

std::size_t peek_offset(std::size_t addr) {
	dd::global_ptr<char> gptr(reinterpret_cast<char*>(
			addr + reinterpret_cast<unsigned long>(startAddr)), false, false);
	return gptr.peek_offset();
}

//CSPext:
argo::node_id_t get_replication_node(std::size_t addr) {
	dd::global_ptr<char> gptr(reinterpret_cast<char*>(
			addr + reinterpret_cast<unsigned long>(startAddr)), false, false);
	return gptr.get_replication_node();
}

std::size_t get_replication_offset(std::size_t addr) {
	dd::global_ptr<char> gptr(reinterpret_cast<char*>(
			addr + reinterpret_cast<unsigned long>(startAddr)), false, false);
	return gptr.get_replication_offset();
}

void load_cache_entry(std::size_t aligned_access_offset) {
	/* CSP: 
	 * We need to refer to the node alternation table when
	 * 	loading from a remote node, in case it's a rebuilt node.
	 * */

	/* If it's not an ArgoDSM address, do not handle it */
	if(aligned_access_offset >= size_of_all){
		return;
	}

	const std::size_t block_size = pagesize*CACHELINE;
	/* Check that the precondition holds true */
	assert((aligned_access_offset % block_size) == 0);

	/* Assign node bit IDs */
	const std::uintptr_t node_id_bit = 1 << getID();
	const std::uintptr_t node_id_inv_bit = ~node_id_bit;

	/* Calculate start values and store some parameters */
	const std::size_t cache_index = getCacheIndex(aligned_access_offset);
	const std::size_t start_index = align_backwards(cache_index, CACHELINE);
	std::size_t end_index = start_index+CACHELINE;
	/* CSP ext: look up alter table. Keep home_node for comparison */
	argo::node_id_t home_node = get_homenode(aligned_access_offset);
	argo::node_id_t load_node = home_node;
	if (useReplication) {
		//home_node = get_homenode(aligned_access_offset);
		load_node = node_alter_tbl[home_node].alter_home_id;
	}
	const std::size_t load_offset = get_offset(aligned_access_offset);
	/* CSP ext: use these temp values to avoid long if-else for ease of coding. */
	MPI_Win real_globalDataWindow = globalDataWindow[home_node];
	std::size_t real_offset = load_offset;	// CSP: dynamic windows may need a special offset

	// CSP: Only one thread at a time can make MPI calls
	sem_wait(&ibsem);

	/* Return if requested cache entry is already up to date. */
	/* CSP: Updated by another thread */
	if(cacheControl[start_index].tag == aligned_access_offset &&
			cacheControl[start_index].state != INVALID){
		sem_post(&ibsem);
		return;
	}

	/* Adjust end_index to ensure the whole chunk to fetch is on the same node */
	for(std::size_t i = start_index+CACHELINE, p = CACHELINE;
					i < start_index+load_size;
					i+=CACHELINE, p+=CACHELINE){
		std::size_t temp_addr = aligned_access_offset + p*block_size;
		/* Increase end_index if it is within bounds and on the same node */
		if(temp_addr < size_of_all && i < cachesize){
			/* CSPext: this temp_node must be mapped to the new home as well */
			/* CSP TODO: 
			 * Should we define the alternative node mapping in peek_homenode, 
			 *  which goes into the definition of global_ptr? Maybe not...?
			 * */
			argo::node_id_t temp_node = peek_homenode(temp_addr);
			if (useReplication) {
				temp_node = node_alter_tbl[peek_homenode(temp_addr)].alter_home_id;
			}
			const std::size_t temp_offset = peek_offset(temp_addr);
			if(temp_node == load_node && temp_offset == (load_offset + p*block_size)){
				end_index+=CACHELINE;
			}else{
				break;
			}
		}else{
			/* Stop when either condition is not satisfied */
			break;
		}
	}

	bool new_sharer = false;
	const std::size_t fetch_size = end_index - start_index;
	const std::size_t classification_size = fetch_size*2;

	/* For each page to load, true if page should be cached else false */
	std::vector<bool> pages_to_load(fetch_size);
	/* For each page to update in the cache, true if page has
	 * already been handled else false */
	std::vector<bool> handled_pages(fetch_size);
	/* Contains classification index for each page to load */
	std::vector<std::size_t> classification_index_array(fetch_size);
	/* Store sharer state from local node temporarily */
	std::vector<std::uintptr_t> local_sharers(fetch_size);
	/* Store content of remote Pyxis directory temporarily */
	std::vector<std::uintptr_t> remote_sharers(classification_size);
	/* Store updates to be made to remote Pyxis directory */
	std::vector<std::uintptr_t> sharer_bit_mask(classification_size);
	/* Temporarily store remotely fetched cache data */
	std::vector<char> temp_data(fetch_size*pagesize);

	/* Write back existing cache entries if needed */
	for(std::size_t idx = start_index, p = 0; idx < end_index; idx+=CACHELINE, p+=CACHELINE){
		/* Address and pointer to the data being loaded */
		const std::size_t temp_addr = aligned_access_offset + p*block_size;

		/* Skip updating pages that are already present and valid in the cache */
		if(cacheControl[idx].tag == temp_addr && cacheControl[idx].state != INVALID){
			pages_to_load[p] = false;
			continue;
		}else{
			pages_to_load[p] = true;
		}

		/* If another page occupies the cache index, begin to evict it. */
		if((cacheControl[idx].tag != temp_addr) && (cacheControl[idx].tag != GLOBAL_NULL)){
			void* old_ptr = static_cast<char*>(startAddr) + cacheControl[idx].tag;
			void* temp_ptr = static_cast<char*>(startAddr) + temp_addr;

			/* If the page is dirty, write it back */
			if(cacheControl[idx].dirty == DIRTY){
				mprotect(old_ptr,block_size,PROT_READ);
				for(std::size_t j=0; j < CACHELINE; j++){
					storepageDIFF(idx+j,pagesize*j+(cacheControl[idx].tag));
				}
				argo_write_buffer->erase(idx);
			}

			/* Ensure the writeback has finished */
			for(int i = 0; i < numtasks; i++){
				if(barwindowsused[i] == 1){
					MPI_Win_unlock(i, globalDataWindow[i]);
					barwindowsused[i] = 0;
				}
			}

			/* Clean up cache and protect memory */
			cacheControl[idx].state = INVALID;
			cacheControl[idx].tag = temp_addr;
			cacheControl[idx].dirty = CLEAN;
			vm::map_memory(temp_ptr, block_size, pagesize*idx, PROT_NONE);
			mprotect(old_ptr,block_size,PROT_NONE);
		}
	}

	/* Initialize classification_index_array */
	for(std::size_t i = 0; i < fetch_size; i+=CACHELINE){
		const std::size_t temp_addr = aligned_access_offset + i*block_size;
		classification_index_array[i] = get_classification_index(temp_addr);
	}

	/* Increase stat counter as load will be performed */
	stats.loads++;

	/* Get globalSharers info from local node and add self to it */
	MPI_Win_lock(MPI_LOCK_SHARED, workrank, 0, sharerWindow);
	for(std::size_t i = 0; i < fetch_size; i+=CACHELINE){
		if(pages_to_load[i]){
			/* Check local pyxis directory if we are sharer of the page */
			local_sharers[i] = (globalSharers[classification_index_array[i]])&node_id_bit;
			if(local_sharers[i] == 0){
				sharer_bit_mask[i*2] = node_id_bit;
				new_sharer = true; //At least one new sharer detected
			}
		}
	}
	MPI_Win_unlock(workrank, sharerWindow);

	/* If this node is a new sharer of at least one of the pages */
	if(new_sharer){
		/* Register this node as sharer of all newly shared pages in the load_node's
		 * globalSharers directory using one MPI call. When this call returns,
		 * remote_sharers contains remote globalSharers directory values prior to
		 * this call. */
		MPI_Win_lock(MPI_LOCK_SHARED, load_node, 0, sharerWindow);
		MPI_Get_accumulate(sharer_bit_mask.data(), classification_size, MPI_LONG,
				remote_sharers.data(), classification_size, MPI_LONG,
				load_node, classification_index_array[0], classification_size,
				MPI_LONG, MPI_BOR, sharerWindow);
		MPI_Win_unlock(load_node, sharerWindow);
	}

	/* Register the received remote globalSharers information locally */
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, sharerWindow);
	for(std::size_t i = 0; i < fetch_size; i+=CACHELINE){
		if(pages_to_load[i]){
			globalSharers[classification_index_array[i]] |= remote_sharers[i*2];
			globalSharers[classification_index_array[i]] |= node_id_bit; //Also add self
			globalSharers[classification_index_array[i]+1] |= remote_sharers[(i*2)+1];
		}
	}
	MPI_Win_unlock(workrank, sharerWindow);

	/* If any owner of a page we loaded needs to downgrade from private
	 * to shared, we need to notify it */
	for(std::size_t i = 0; i < fetch_size; i+=CACHELINE){
		/* Skip pages that are not loaded or already handled */
		if(pages_to_load[i] && !handled_pages[i]){
			std::fill(sharer_bit_mask.begin(), sharer_bit_mask.end(), 0);
			const std::uintptr_t owner_id_bit =
				remote_sharers[i*2]&node_id_inv_bit; // remove own bit

			/* If there is exactly one other owner, and we are not sharer */
			if(isPowerOf2(owner_id_bit) && owner_id_bit != 0 && local_sharers[i] == 0){
				std::uintptr_t owner = invalid_node; // initialize to failsafe value
				for(int n = 0; n < numtasks; n++) {
					if(1ul<<n==owner_id_bit) {
						owner = n; //just get rank...
						break;
					}
				}
				sharer_bit_mask[i*2] = node_id_bit;

				/* Check if any of the remaining pages need downgrading on the same node */
				for(std::size_t j = i+CACHELINE; j < fetch_size; j+=CACHELINE){
					if(pages_to_load[j] && !handled_pages[j]){
						if((remote_sharers[j*2]&node_id_inv_bit) == owner_id_bit &&
								local_sharers[j] == 0){
							sharer_bit_mask[j*2] = node_id_bit;
							handled_pages[j] = true; //Ensure these are marked as completed
						}
					}
				}

				/* Downgrade all relevant pages on the owner node from private to shared */
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, owner, 0, sharerWindow);
				MPI_Accumulate(sharer_bit_mask.data(), classification_size, MPI_LONG, owner,
						classification_index_array[0], classification_size, MPI_LONG,
						MPI_BOR, sharerWindow);
				MPI_Win_unlock(owner, sharerWindow);
			}
		}
	}

	/* Finally, get the cache data and store it temporarily */

	/* CSPext: Add a branch here to check if the home node is down */
	if (load_node != home_node) {
		/* CSP: load_node points to alternative node, need to use different window */
		local_check_after_recovery(&(node_alter_tbl[home_node]));
		if (load_node == -1) {
			/* CSP: using EC & redirect. Just go to repl page */
			load_node = get_replication_node(aligned_access_offset);
			real_offset = get_replication_offset(aligned_access_offset);
		} else {
			/* CSP: using rebuild, or using CR & redirect. Look up the table. */
			real_globalDataWindow = node_alter_tbl[home_node].alter_globalDataWindow;
			if (env::replication_recovery_policy() == 1) {
				/* CSP: dynamic windows used in policy 1 (rebuild) needs extra offset. */
				// CSP TODO: Use MPI_Get_address on origin node instead of casting to MPI_Aint/size_t!
				real_offset += (size_t)node_alter_tbl[home_node].alter_globalData;
			}
		}
	}
	MPI_Win_lock(MPI_LOCK_SHARED, load_node , 0, real_globalDataWindow);
	MPI_Get(temp_data.data(), fetch_size, cacheblock,
			load_node, real_offset, 
			fetch_size, cacheblock, real_globalDataWindow);
	MPI_Win_unlock(load_node, real_globalDataWindow);

	/* Update the cache */
	for(std::size_t idx = start_index, p = 0; idx < end_index; idx+=CACHELINE, p+=CACHELINE){
		/* Update only the pages necessary */
		if(pages_to_load[p]){
			/* Insert the data in the node cache */
			memcpy(&cacheData[idx*block_size], &temp_data[p*block_size], block_size);

			const std::size_t temp_addr = aligned_access_offset + p*block_size;
			void* temp_ptr = static_cast<char*>(startAddr) + temp_addr;

			/* If this is the first time inserting in to this index, perform vm map */
			if(cacheControl[idx].tag == GLOBAL_NULL){
				vm::map_memory(temp_ptr, block_size, pagesize*idx, PROT_READ);
				cacheControl[idx].tag = temp_addr;
			}else{
				/* Else, just mprotect the region */
				mprotect(temp_ptr, block_size, PROT_READ);
			}
			touchedcache[idx] = 1;
			cacheControl[idx].state = VALID;
			cacheControl[idx].dirty=CLEAN;
		}
	}
	sem_post(&ibsem);
}


void initmpi(){
	int ret,initialized,thread_status;
	int thread_level = (ARGO_ENABLE_MT == 1) ? MPI_THREAD_MULTIPLE : MPI_THREAD_SERIALIZED;
	MPI_Initialized(&initialized);
	if (!initialized){
		ret = MPI_Init_thread(NULL,NULL,thread_level,&thread_status);
	}
	else{
		printf("MPI was already initialized before starting ArgoDSM - shutting down\n");
		exit(EXIT_FAILURE);
	}

	if (ret != MPI_SUCCESS || thread_status != thread_level) {
		printf ("MPI not able to start properly\n");
		MPI_Abort(MPI_COMM_WORLD, ret);
		exit(EXIT_FAILURE);
	}

	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	init_mpi_struct();
	init_mpi_cacheblock();
}

argo::node_id_t getID(){
	return workrank;
}
argo::node_id_t argo_get_nid(){
	return workrank;
}

/* CSPext: Wrapping up a function to calculate the replication node */
argo::node_id_t argo_get_rid(){
	/* CSPext: Does not need to return invalid_node_id 
	 * 	because only an unassigned page returns a node_id */
	return (workrank + 1) % argo_get_nodes();
}

/* CSPext: A function to calculate the replication node */
argo::node_id_t argo_calc_rid(argo::node_id_t n){
	if (n < 0) {
		return dd::invalid_node_id;
	} else {
		return (n + 1) % argo_get_nodes();
	}
}

unsigned int argo_get_nodes(){
	return numtasks;
}
unsigned int getThreadCount(){
	return NUM_THREADS;
}

//My sort of allocatefunction now since parmacs macros had this design
void * argo_gmalloc(unsigned long size){
	if(argo_get_nodes()==1){return malloc(size);}

	pthread_mutex_lock(&gmallocmutex);
	MPI_Barrier(workcomm);

	unsigned long roundedUp; //round up to number of pages to use.
	unsigned long currPage; //what pages has been allocated previously
	unsigned long alignment = pagesize*CACHELINE;

	roundedUp = size/(alignment);
	roundedUp = (alignment)*(roundedUp+1);

	currPage = (*allocationOffset)/(alignment);
	currPage = (alignment) *(currPage);

	if((*allocationOffset) +size > size_of_all){
		pthread_mutex_unlock(&gmallocmutex);
		return NULL;
	}

	void *ptrtmp = (char*)startAddr+*allocationOffset;
	*allocationOffset = (*allocationOffset) + roundedUp;

	if(ptrtmp == NULL){
		pthread_mutex_unlock(&gmallocmutex);
		exit(EXIT_FAILURE);
	}
	else{
		memset(ptrtmp,0,roundedUp);
	}
	swdsm_argo_barrier(1);
	pthread_mutex_unlock(&gmallocmutex);
	return ptrtmp;
}

/**
 * @brief aligns an offset (into a memory region) to the beginning of its
 * subsequent size block if it is not already aligned to a size block.
 * @param offset the unaligned offset
 * @param size the size of each block
 * @return the aligned offset
 */
std::size_t align_forwards(std::size_t offset, std::size_t size){
	return (offset == 0) ? offset : (1 + ((offset-1) / size))*size;
}

void argo_initialize(std::size_t argo_size, std::size_t cache_size){
	int i;
	unsigned long j;
	initmpi();

	/** Standardise the ArgoDSM memory space */
	argo_size = std::max(argo_size, static_cast<std::size_t>(pagesize*numtasks));
	argo_size = align_forwards(argo_size, pagesize*CACHELINE*numtasks*dd::policy_padding());

	startAddr = vm::start_address();
#ifdef ARGO_PRINT_STATISTICS
	printf("maximum virtual memory: %ld GiB\n", vm::size() >> 30);
#endif

	threadbarrier = (pthread_barrier_t *) malloc(sizeof(pthread_barrier_t)*(NUM_THREADS+1));
	for(i = 1; i <= NUM_THREADS; i++){
		pthread_barrier_init(&threadbarrier[i],NULL,i);
	}

	/** Get the number of pages to load from the env module */
	load_size = env::load_size();
	/** Limit cache_size to at most argo_size */
	cachesize = std::min(argo_size, cache_size);
	/** Round the number of cache pages upwards */
	cachesize = align_forwards(cachesize, pagesize*CACHELINE);
	/** At least two pages are required to prevent endless eviction loops */
	cachesize = std::max(cachesize, static_cast<unsigned long>(pagesize*CACHELINE*2));
	cachesize /= pagesize;

	classificationSize = 2*(argo_size/pagesize);
	argo_write_buffer = new write_buffer<std::size_t>();

	barwindowsused = (char *)malloc(numtasks*sizeof(char));
	for(i = 0; i < numtasks; i++){
		barwindowsused[i] = 0;
	}

	int *workranks = (int *) malloc(sizeof(int)*numtasks);
	int *procranks = (int *) malloc(sizeof(int)*2);
	int workindex = 0;

	for(i = 0; i < numtasks; i++){
		workranks[workindex++] = i;
		procranks[0]=i;
		procranks[1]=i+1;
	}

	MPI_Comm_group(MPI_COMM_WORLD, &startgroup);
	MPI_Group_incl(startgroup,numtasks,workranks,&workgroup);
	MPI_Comm_create(MPI_COMM_WORLD,workgroup,&workcomm);
	MPI_Group_rank(workgroup,&workrank);


	//Allocate local memory for each node,
	size_of_all = argo_size; //total distr. global memory
	GLOBAL_NULL=size_of_all+1;
	size_of_chunk = argo_size/(numtasks); //part on each node
	// CSPext
	useReplication = argo_get_nodes() > 1 && env::replication_policy() != 0;
	if (useReplication) {
		if (env::replication_policy() == 1) {
			// complete replication
			printf("COMPLETE REPLICATION\n");
			size_of_replication = size_of_chunk;
		}
		else if (env::replication_policy() == 2) {
			// erasure coding (n-1, 1)
			printf("ERASURE CODING - data fragments: %lu, parity fragments: %lu\n", env::replication_data_fragments(), env::replication_parity_fragments());
			size_of_replication = size_of_chunk / (env::replication_data_fragments());
			size_of_replication = ((size_of_replication / pagesize) + 1) * pagesize; // align with pagesize
		}
	}
	// CSPext: initialize a table for pages that has no repl (for EC + redirect)
	no_repl_page = (bool *)malloc(sizeof(bool) * (size_of_all / pagesize) + 1);
	for (i = 0; (unsigned long)i < (size_of_all / pagesize) + 1; ++i) {
		no_repl_page[i] = false;
	}
	

	sig::signal_handler<SIGSEGV>::install_argo_handler(&handler);

	unsigned long cacheControlSize = sizeof(control_data)*cachesize;
	unsigned long gwritersize = classificationSize*sizeof(long);
	cacheControlSize = align_forwards(cacheControlSize, pagesize);
	gwritersize = align_forwards(gwritersize, pagesize);

	owners_dir_size = 3*(argo_size/pagesize);
	std::size_t owners_dir_size_bytes = owners_dir_size*sizeof(std::size_t);
	owners_dir_size_bytes = align_forwards(owners_dir_size_bytes, pagesize);

	std::size_t offsets_tbl_size = numtasks;
	std::size_t offsets_tbl_size_bytes = offsets_tbl_size*sizeof(std::size_t);
	offsets_tbl_size_bytes = align_forwards(offsets_tbl_size_bytes, pagesize);

	cacheoffset = pagesize*cachesize+cacheControlSize;

	globalData = static_cast<char*>(vm::allocate_mappable(pagesize, size_of_chunk));
	// CSPext: 
	if (useReplication) {
		replData = static_cast<char*>(vm::allocate_mappable(pagesize, size_of_replication));
	}
	cacheData = static_cast<char*>(vm::allocate_mappable(pagesize, cachesize*pagesize));
	cacheControl = static_cast<control_data*>(vm::allocate_mappable(pagesize, cacheControlSize));

	touchedcache = (argo_byte *)malloc(cachesize);
	if(touchedcache == NULL){
		printf("malloc error out of memory\n");
		exit(EXIT_FAILURE);
	}

	lockbuffer = static_cast<unsigned long*>(vm::allocate_mappable(pagesize, pagesize));
	pagecopy = static_cast<char*>(vm::allocate_mappable(pagesize, cachesize*pagesize));
	globalSharers = static_cast<unsigned long*>(vm::allocate_mappable(pagesize, gwritersize));

	/* CSPext: Initialize home alternation table */
	if (useReplication) {
		node_alter_tbl = static_cast<node_alternation_table*>(vm::allocate_mappable(pagesize, pagesize));
	}

	if (dd::is_first_touch_policy()) {
		global_owners_dir = static_cast<std::uintptr_t*>(vm::allocate_mappable(pagesize, owners_dir_size_bytes));
		global_offsets_tbl = static_cast<std::uintptr_t*>(vm::allocate_mappable(pagesize, offsets_tbl_size_bytes));
	}

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	MPI_Barrier(MPI_COMM_WORLD);

	void* tmpcache;
	tmpcache=cacheData;
	vm::map_memory(tmpcache, pagesize*cachesize, 0, PROT_READ|PROT_WRITE);

	std::size_t current_offset = pagesize*cachesize;
	tmpcache=cacheControl;
	vm::map_memory(tmpcache, cacheControlSize, current_offset, PROT_READ|PROT_WRITE);

	current_offset += cacheControlSize;
	tmpcache=globalData;
	vm::map_memory(tmpcache, size_of_chunk, current_offset, PROT_READ|PROT_WRITE);

	/* CSPext: map the memory in replData area */
	if (useReplication) {
		current_offset += size_of_chunk;
		tmpcache=replData;
		vm::map_memory(tmpcache, size_of_replication, current_offset, PROT_READ|PROT_WRITE);
		current_offset += size_of_replication;
	}
	else {
		current_offset += size_of_chunk;
	}

	tmpcache=globalSharers;
	vm::map_memory(tmpcache, gwritersize, current_offset, PROT_READ|PROT_WRITE);

	current_offset += gwritersize;
	tmpcache=lockbuffer;
	vm::map_memory(tmpcache, pagesize, current_offset, PROT_READ|PROT_WRITE);

	/* CSPext: remember the offset for future rebuilding */
	if (useReplication) {
		vm_map_offset_record = current_offset + pagesize;
	}

	/* CSPext: map the memory for home node alternation table */
	if (useReplication) {
		current_offset += pagesize;
		tmpcache=node_alter_tbl;
		vm::map_memory(tmpcache, pagesize, current_offset, PROT_READ|PROT_WRITE);
		// CSP: keep track of the offset
		vm_map_offset_record = current_offset + pagesize;
	}

	if (dd::is_first_touch_policy()) {
		current_offset += pagesize;
		tmpcache=global_owners_dir;
		vm::map_memory(tmpcache, owners_dir_size_bytes, current_offset, PROT_READ|PROT_WRITE);
		current_offset += owners_dir_size_bytes;
		tmpcache=global_offsets_tbl;
		vm::map_memory(tmpcache, offsets_tbl_size_bytes, current_offset, PROT_READ|PROT_WRITE);
		// CSP: keep track of the offset
		if (useReplication) {
			vm_map_offset_record = current_offset + offsets_tbl_size_bytes;
		}
	}

	sem_init(&ibsem,0,1);
	sem_init(&globallocksem,0,1);

	allocationOffset = (unsigned long *)calloc(1,sizeof(unsigned long));
	globalDataWindow = (MPI_Win*)malloc(sizeof(MPI_Win)*numtasks);

	for(i = 0; i < numtasks; i++){
 		MPI_Win_create(globalData, size_of_chunk*sizeof(argo_byte), 1,
									 MPI_INFO_NULL, MPI_COMM_WORLD, &globalDataWindow[i]);
	}

	if (useReplication) {
		/* CSPext: initialize replDataWindow */
		replDataWindow = (MPI_Win*)malloc(sizeof(MPI_Win)*numtasks);

		for(i = 0; i < numtasks; i++){
			// CSP: Can potentially optimise by specifying the "no locks" key.
			MPI_Win_create(replData, size_of_replication*sizeof(argo_byte), 1,
									MPI_INFO_NULL, MPI_COMM_WORLD, &replDataWindow[i]);
		}
	}

	MPI_Win_create(globalSharers, gwritersize, sizeof(unsigned long),
								 MPI_INFO_NULL, MPI_COMM_WORLD, &sharerWindow);
	MPI_Win_create(lockbuffer, pagesize, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &lockWindow);

	/* CSPext: initialize node_alter_tbl_window */
	if (useReplication) {
		node_alter_tbl_window = (MPI_Win*)malloc(sizeof(MPI_Win)*numtasks);

		for(i = 0; i < numtasks; i++) {
			MPI_Win_create(node_alter_tbl, pagesize*sizeof(argo_byte), 1,
									MPI_INFO_NULL, MPI_COMM_WORLD, &node_alter_tbl_window[i]);
		}
	}

	if (dd::is_first_touch_policy()) {
		MPI_Win_create(global_owners_dir, owners_dir_size_bytes, sizeof(std::uintptr_t),
									 MPI_INFO_NULL, MPI_COMM_WORLD, &owners_dir_window);
		MPI_Win_create(global_offsets_tbl, offsets_tbl_size_bytes, sizeof(std::uintptr_t),
									 MPI_INFO_NULL, MPI_COMM_WORLD, &offsets_tbl_window);
	}

	memset(pagecopy, 0, cachesize*pagesize);
	memset(touchedcache, 0, cachesize);
	memset(globalData, 0, size_of_chunk*sizeof(argo_byte));
	if (useReplication) {
		/* CSPext: initialize replData to all 0 */
		memset(replData, 0, size_of_replication*sizeof(argo_byte));
	}
	memset(cacheData, 0, cachesize*pagesize);
	memset(lockbuffer, 0, pagesize);
	memset(globalSharers, 0, gwritersize);
	memset(cacheControl, 0, cachesize*sizeof(control_data));
	/* CSPext: initialize home_alter_tbl */
	if (useReplication) {
		for (i = 0; i < numtasks; i++) {
			/* CSP: alter_id starts with i, meaning no alternation */
			node_alter_tbl[i].alter_home_id = i;
			node_alter_tbl[i].alter_repl_id = i;
			/* CSP: will be used for recovery policy 1 */
			node_alter_tbl[i].alter_globalData = NULL;
			node_alter_tbl[i].alter_replData = NULL;

			if (env::replication_recovery_policy() == 1) {
				/* CSP: For rebuilding policy, windows must be created beforehand
				 * But the memory location is not determined. We use dynamic 
				 *   windows and delay attachment of memory till when the memory
				 *   is created
				 */
				MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, 
								&(node_alter_tbl[i].alter_globalDataWindow));
				MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, 
								&(node_alter_tbl[i].alter_replDataWindow));
			} else {
				/* CSP: otherwise just initialize it to NULL */
				node_alter_tbl[i].alter_globalDataWindow = NULL;
				node_alter_tbl[i].alter_replDataWindow = NULL;
			}

			// CSP: a flag to create an globalDataWindow
			node_alter_tbl[i].just_recovered = false;
			node_alter_tbl[i].refresh_replDataWindow = false;

			// CSP: initialize recovery blocker lock
			node_alter_tbl[i].recovering = false;
		}
	}

	if (dd::is_first_touch_policy()) {
		memset(global_owners_dir, 0, owners_dir_size_bytes);
		memset(global_offsets_tbl, 0, offsets_tbl_size_bytes);
	}

	for(j=0; j<cachesize; j++){
		cacheControl[j].tag = GLOBAL_NULL;
		cacheControl[j].state = INVALID;
		cacheControl[j].dirty = CLEAN;
	}

	argo_reset_coherence(1);
}

void argo_finalize(){
	int i;
	swdsm_argo_barrier(1);
	if(getID() == 0){
		printf("ArgoDSM shutting down\n");
	}
	swdsm_argo_barrier(1);
	mprotect(startAddr,size_of_all,PROT_WRITE|PROT_READ);
	MPI_Barrier(MPI_COMM_WORLD);

	for(i=0; i <numtasks;i++){
		if(i==workrank){
			printStatistics();
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	for(i=0; i<numtasks; i++){
		MPI_Win_free(&globalDataWindow[i]);
		if (useReplication) {
			/* CSPext: free replDataWindow */
			MPI_Win_free(&replDataWindow[i]);
			/* CSPext: free node_alter_tbl_window */
			MPI_Win_free(&node_alter_tbl_window[i]);
			if (env::replication_recovery_policy() == 1) {
				/* CSPext: free alter_globalDataWindow in node_alter_tbl */
				/* CSP: freeing the window automatically detatches all memory */
				MPI_Win_free(&(node_alter_tbl[i].alter_globalDataWindow));
				MPI_Win_free(&(node_alter_tbl[i].alter_replDataWindow));
			}
		}
	}
	MPI_Win_free(&sharerWindow);
	MPI_Win_free(&lockWindow);
	if (dd::is_first_touch_policy()) {
		MPI_Win_free(&owners_dir_window);
		MPI_Win_free(&offsets_tbl_window);
	}
	MPI_Comm_free(&workcomm);
	MPI_Finalize();
	return;
}

void self_invalidation(){
	unsigned long i;
	double t1,t2;
	int flushed = 0;
	unsigned long id = 1 << getID();

	t1 = MPI_Wtime();
	for(i = 0; i < cachesize; i+=CACHELINE){
		if(touchedcache[i] != 0){
			unsigned long distrAddr = cacheControl[i].tag;
			unsigned long lineAddr = distrAddr/(CACHELINE*pagesize);
			lineAddr*=(pagesize*CACHELINE);
			unsigned long classidx = get_classification_index(lineAddr);
			argo_byte dirty = cacheControl[i].dirty;

			if(flushed == 0 && dirty == DIRTY){
				argo_write_buffer->flush();
				flushed = 1;
			}
			MPI_Win_lock(MPI_LOCK_SHARED, workrank, 0, sharerWindow);
			if(
				 // node is single writer
				 (globalSharers[classidx+1]==id)
				 ||
				 // No writer and assert that the node is a sharer
				 ((globalSharers[classidx+1]==0) && ((globalSharers[classidx]&id)==id))
				 ){
				MPI_Win_unlock(workrank, sharerWindow);
				touchedcache[i] =1;
				/*nothing - we keep the pages, SD is done in flushWB*/
			}
			else{ //multiple writer or SO
				MPI_Win_unlock(workrank, sharerWindow);
				cacheControl[i].dirty=CLEAN;
				cacheControl[i].state = INVALID;
				touchedcache[i] =0;
				mprotect((char*)startAddr + lineAddr, pagesize*CACHELINE, PROT_NONE);
			}
		}
	}
	t2 = MPI_Wtime();
	stats.selfinvtime += (t2-t1);
}

void swdsm_argo_barrier(int n){ //BARRIER
	double time1,time2;
	pthread_t barrierlockholder;
	time1 = MPI_Wtime();
	pthread_barrier_wait(&threadbarrier[n]);
	if(argo_get_nodes()==1){
		time2 = MPI_Wtime();
		stats.barriers++;
		stats.barriertime += (time2-time1);
		return;
	}

	if(pthread_mutex_trylock(&barriermutex) == 0){
		barrierlockholder = pthread_self();
		pthread_mutex_lock(&cachemutex);
		sem_wait(&ibsem);
		argo_write_buffer->flush();
		MPI_Barrier(workcomm);
		self_invalidation();
		sem_post(&ibsem);
		pthread_mutex_unlock(&cachemutex);
	}

	pthread_barrier_wait(&threadbarrier[n]);
	if(pthread_equal(barrierlockholder,pthread_self())){
		pthread_mutex_unlock(&barriermutex);
		time2 = MPI_Wtime();
		stats.barriers++;
		stats.barriertime += (time2-time1);
	}
}

void argo_reset_coherence(int n){
	unsigned long j;
	stats.writebacks = 0;
	stats.stores = 0;
	memset(touchedcache, 0, cachesize);

	sem_wait(&ibsem);
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, sharerWindow);
	for(j = 0; j < classificationSize; j++){
		globalSharers[j] = 0;
	}
	MPI_Win_unlock(workrank, sharerWindow);
	
	if (dd::is_first_touch_policy()) {
		/**
		 * @note initialize the first-touch directory with a magic value,
		 *       in order to identify if the indices are touched or not.
		 */
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, owners_dir_window);
		for(j = 0; j < owners_dir_size; j++) {
			global_owners_dir[j] = GLOBAL_NULL;
		}
		MPI_Win_unlock(workrank, owners_dir_window);

		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, offsets_tbl_window);
		for(j = 0; j < static_cast<std::size_t>(numtasks); j++) {
			global_offsets_tbl[j] = 0;
		}
		MPI_Win_unlock(workrank, offsets_tbl_window);
	}
	sem_post(&ibsem);
	swdsm_argo_barrier(n);
	mprotect(startAddr,size_of_all,PROT_NONE);
	swdsm_argo_barrier(n);
	clearStatistics();
}

void argo_acquire(){
	int flag;
	pthread_mutex_lock(&cachemutex);
	sem_wait(&ibsem);
	self_invalidation();
	MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,workcomm,&flag,MPI_STATUS_IGNORE);
	sem_post(&ibsem);
	pthread_mutex_unlock(&cachemutex);
}


void argo_release(){
	int flag;
	pthread_mutex_lock(&cachemutex);
	sem_wait(&ibsem);
	argo_write_buffer->flush();
	MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,workcomm,&flag,MPI_STATUS_IGNORE);
	sem_post(&ibsem);
	pthread_mutex_unlock(&cachemutex);
}

void argo_acq_rel(){
	argo_acquire();
	argo_release();
}

double argo_wtime(){
	return MPI_Wtime();
}

void clearStatistics(){
	stats.selfinvtime = 0;
	stats.loadtime = 0;
	stats.storetime = 0;
	stats.flushtime = 0;
	stats.writebacktime = 0;
	stats.locktime=0;
	stats.barriertime = 0;
	stats.stores = 0;
	stats.writebacks = 0;
	stats.loads = 0;
	stats.barriers = 0;
	stats.locks = 0;
	stats.ssitime = 0;
	stats.ssdtime = 0;
}

void storepageDIFF(unsigned long index, unsigned long addr){
	unsigned int i,j;
	int cnt = 0;
	const argo::node_id_t homenode = get_homenode(addr);
	const std::size_t offset = get_offset(addr);
	
	// CSPext: for replication (initialize later)
	argo::node_id_t repl_node = 0;
	std::size_t repl_offset = 0;


	// printf("----storepagediff: Node = %d\n", argo_get_nid());

	char * copy = (char *)(pagecopy + index*pagesize);
	char * real = (char *)startAddr+addr;
	size_t drf_unit = sizeof(char);

	/* CSPext: Pointer to actually used MPI windows. */
	MPI_Win real_globalDataWindow = globalDataWindow[homenode];
	MPI_Win real_replDataWindow = NULL; // CSP: Init later
	argo::node_id_t real_home_id = homenode; // CSP: May change later
	argo::node_id_t real_repl_id = 0; // CSP: Init later
	/* CSP:
	 * For any operation on a window initialized by MPI_Win_create_dynamic(),
	 *  always set the target_disp as the true virtual addr on the origin node.
	 * We use dynamic windows under recovery policy 1; in that case these
	 *  offsets will have an increment (see below).
	 * Or if we use EC + redirect, we'll need look for a different offset.
	 * */
	size_t real_home_offset = offset;
	size_t real_repl_offset = repl_offset;
	
	/* CSPext: initialize repl related vars only if useReplication */
	if (useReplication) {
		repl_node = get_replication_node(addr);
		repl_offset = get_replication_offset(addr);
		real_repl_offset = repl_offset;
		real_replDataWindow = replDataWindow[repl_node];
		real_home_id = node_alter_tbl[homenode].alter_home_id;
		real_repl_id = node_alter_tbl[repl_node].alter_repl_id;
		
		/* CSPext: special check: with EC + redirect, be careful about repl pages */
		if (env::replication_policy() == 2 
				&& env::replication_recovery_policy() == 0) {
			if (no_repl_page[addr / size_of_all]) {
				real_repl_id = -1;	// skip repl writing of this page
			}
		}

		if (real_home_id != homenode) { 
			/* CSP: home node is down*/

			/* CSP: this function below fixes the datawindow and resets refresh flag.
			 * This is because sometimes (policy 0) the window has to be set locally;
			 *  the remote node writing the node alternation table did not 
			 *  know the value of the window.
			 * */
			local_check_after_recovery(&(node_alter_tbl[homenode]));
			real_globalDataWindow = node_alter_tbl[homenode].alter_globalDataWindow;
			if (env::replication_recovery_policy() == 1) {
				/* CSP: Under recovery policy 1 (rebuild), we use dynamic windows.
				 * It requires the offset to be relative to the start of a machine's
				 * 	virtual address. 
				 * Here we set offset increment (normally 0 --no increment) to the 
				 *  start of the memory area pointed to by the window on target node.
				 * */
				// CSP TODO: Use MPI_Get_address on origin node instead of casting to MPI_Aint!
				real_home_offset =+ (size_t)node_alter_tbl[homenode].alter_globalData;
			} else if (env::replication_recovery_policy() == 0) {	
				/* CSP: Under recovery policy 0 (redirect), two nodes no longer have repl:
				 * 	1. the dead node, on which the repl area is lost (id set to -1);
				 * 	2. the dead node's repl node, on which the repl area is now the 
				 * 		alternative data area for dead node
				 *
				 * In this branch we detect scenario 2. We forbid writing repl data
				 *  by setting real_repl_id to -1 --indicating it's not writable.
				 * */
				real_repl_id = -1;
				if (env::replication_policy() == 2) {
					/* CSP: EC + rebuilding, have to search for target page */
					real_home_id = get_replication_node(addr);
					real_home_offset = get_replication_offset(addr);
				}
			}
		}
		if (real_repl_id != repl_node && real_repl_id >= 0) {
			/* CSP: a different alter_repl_id value indicates old repl_id is dead;
			 *  or, real_repl_id == -1 indicates writing repl data is skipped 
			 * */
			//node_alter_tbl_create_replDatawindow(&(node_alter_tbl[repl_node]));
			real_replDataWindow = node_alter_tbl[repl_node].alter_replDataWindow;
			if (env::replication_recovery_policy() == 1) {
				real_repl_offset = (size_t)node_alter_tbl[repl_node].alter_replData;
			}
		}
	}

	/* CSP ext: locking action is different when using alternative node. */
	if (useReplication && real_home_id != homenode) {
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, real_home_id, 0, real_globalDataWindow);
	} else {
		if(barwindowsused[homenode] == 0){
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, homenode, 0, globalDataWindow[homenode]);
			barwindowsused[homenode] = 1;
		}
	}

	if (useReplication && real_repl_id >= 0) {
		/* CSP ext: lock repl data window as well */
		//fprintf(stderr, "%d: ----check: gwindow: %p rwindow: real_repl_id: %d\n", argo_get_nid(), real_globalDataWindow, real_repl_id);
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, real_repl_id, 0, real_replDataWindow);
	}

	for(i = 0; i < pagesize; i+=drf_unit){
		int branchval;
		for(j=i; j < i+drf_unit; j++){
			branchval = real[j] != copy[j];
			if(branchval != 0){
				break;
			}
		}
		if(branchval != 0){
			cnt+=drf_unit;
		}
		else{
			if(cnt > 0){
				//fprintf(stderr, "%d: writing to %d in store to %p\n", argo_get_nid(), real_home_id, real_globalDataWindow);
				MPI_Put(&real[i-cnt], cnt, MPI_BYTE, real_home_id, real_home_offset+(i-cnt), cnt, MPI_BYTE, real_globalDataWindow);
				//fprintf(stderr, "%d: finished one writing in store\n", argo_get_nid());
				// CSPext: Update page on repl node
				if (useReplication) {
					if (env::replication_policy() == 1 && real_repl_id >= 0) {
						MPI_Put(&real[i-cnt], cnt, MPI_BYTE, real_repl_id, real_repl_offset+(i-cnt), cnt, MPI_BYTE, real_replDataWindow);
					}
					else if (env::replication_policy() == 2 && real_repl_id >= 0) {
						MPI_Accumulate(&real[i-cnt], cnt, MPI_BYTE, real_repl_id, real_repl_offset+(i-cnt), cnt, MPI_BYTE, MPI_BXOR, real_replDataWindow);
					}
				}
				cnt = 0;
			}
		}
	}
	if(cnt > 0){
		MPI_Put(&real[i-cnt], cnt, MPI_BYTE, real_home_id, real_home_offset+(i-cnt), cnt, MPI_BYTE, real_globalDataWindow);
		// CSPext: Update page on repl node
		if (useReplication) {		
			if (env::replication_policy() == 1 && real_repl_id >= 0) {
				MPI_Put(&real[i-cnt], cnt, MPI_BYTE, real_repl_id, real_repl_offset+(i-cnt), cnt, MPI_BYTE, real_replDataWindow);
			}
			else if (env::replication_policy() == 2 && real_repl_id >= 0) {
				MPI_Accumulate(&real[i-cnt], cnt, MPI_BYTE, real_repl_id, real_repl_offset+(i-cnt), cnt, MPI_BYTE, MPI_BXOR, real_replDataWindow);
			}
		}
	}
	stats.stores++;
	
	if (useReplication && real_repl_id >= 0) {
		/* CSPext: Unlock repl node window */
		MPI_Win_unlock(real_repl_id, real_replDataWindow);
	}
	/* CSPext: need to unlock different window for an alternated node */
	if (useReplication && real_home_id != homenode) {
		MPI_Win_unlock(real_home_id, real_globalDataWindow);
	}
}

void printStatistics(){
	printf("#####################STATISTICS#########################\n");
	printf("# PROCESS ID %d \n",workrank);
	printf("cachesize:%ld,CACHELINE:%ld wbsize:%ld\n",cachesize,CACHELINE,
			env::write_buffer_size()/CACHELINE);
	printf("     writebacktime+=(t2-t1): %lf\n",stats.writebacktime);
	printf("# Storetime : %lf , loadtime :%lf flushtime:%lf, writebacktime: %lf\n",
			stats.storetime, stats.loadtime, stats.flushtime, stats.writebacktime);
	printf("# SSDtime:%lf, SSItime:%lf\n", stats.ssdtime, stats.ssitime);
	printf("# Barriertime : %lf, selfinvtime %lf\n",stats.barriertime, stats.selfinvtime);
	printf("stores:%lu, loads:%lu, barriers:%lu\n",stats.stores,stats.loads,stats.barriers);
	printf("Locks:%d\n",stats.locks);
	printf("########################################################\n");
	printf("\n\n");
}

void *argo_get_global_base(){return startAddr;}
size_t argo_get_global_size(){return size_of_all;}

unsigned long get_classification_index(uint64_t addr){
	return (2*(addr/(pagesize*CACHELINE))) % classificationSize;
}

bool _is_cached(std::size_t addr) {
	argo::node_id_t homenode;
	std::size_t aligned_address = align_backwards(
			addr-reinterpret_cast<std::size_t>(startAddr), CACHELINE*pagesize);
	homenode = peek_homenode(aligned_address);
	std::size_t cache_index = getCacheIndex(aligned_address);

	// Return true for pages which are either local or already cached
	return ((homenode == getID()) || (cacheControl[cache_index].tag == aligned_address &&
				cacheControl[cache_index].state == VALID));
}

/* CSPext: Node rebuild function
 * 
 * If REPLICATION_RECOVERY_POLICY is 1, this function rebuilds the lost data.
 * Else if REPLICATION_RECOVERY_POLICY is 0, this function modifies the node
 * 	alternation table to redirect all accesses to the dead node.
 *
 * Plus, the function runs differently for complete replication and EC.
 * */
void lost_node_data_recovery(argo::node_id_t dead_node) {

	if (argo_get_nid() == dead_node || !useReplication) {
		return;
	}

	/* CSP: MPI window offsets (for writing to remote node) */
	const char *tbl_base_addr = (char *)(&(node_alter_tbl[dead_node]));
	const size_t alter_hid_offset
			= (char *)(&(node_alter_tbl[dead_node].alter_home_id)) - tbl_base_addr;
	const size_t alter_gdata_offset
			= (char *)(&(node_alter_tbl[dead_node].alter_globalData)) - tbl_base_addr; 
	//const size_t alter_gWindow_offset
	//		= (char *)(&(node_alter_tbl[dead_node].alter_globalDataWindow)) - tbl_base_addr; 
	const size_t just_recovered_offset
			= (char *)(&(node_alter_tbl[dead_node].just_recovered)) - tbl_base_addr; 
	const size_t alter_rid_offset
			= (char *)(&(node_alter_tbl[dead_node].alter_repl_id)) - tbl_base_addr; 
	const size_t alter_rdata_offset
			= (char *)(&(node_alter_tbl[dead_node].alter_replData)) - tbl_base_addr; 
	//const size_t alter_rWindow_offset
	//		= (char *)(&(node_alter_tbl[dead_node].alter_replDataWindow)) - tbl_base_addr; 
	//const size_t refresh_rWindow_offset
	//		= (char *)(&(node_alter_tbl[dead_node].refresh_replDataWindow)) - tbl_base_addr; 
	const size_t recovering_offset 
			= (char *)(&(node_alter_tbl[dead_node].recovering)) - tbl_base_addr;

	/* CSP: Data to write */
	node_alternation_table temp_tbl;		// CSP: temparary var for MPI_Put
	
	/* CSP: Variables for rebuild locking */
	const bool original_bool = true; 	// CSP: Swap this into the origin
	const bool compare_bool = false;	// CSP: Compare to this
	bool result_bool = true;			// CSP: Swap the origin into this
	bool finished_work = false;
	
	/* CSP: find the (id of) source of data to recover lost data pieces. */
	argo::node_id_t repl_node = argo_calc_rid(dead_node);
	MPI_Win gdata_source_win 
			= ((repl_node == node_alter_tbl[repl_node].alter_repl_id) ? 
				replDataWindow[repl_node] : 
				node_alter_tbl[repl_node].alter_replDataWindow);
	repl_node = node_alter_tbl[repl_node].alter_repl_id;
	argo::node_id_t home_of_rdata = ((dead_node == 0) ? (argo_get_nodes() - 1) : (dead_node - 1));
	MPI_Win rdata_source_win 
			= ((home_of_rdata == node_alter_tbl[home_of_rdata].alter_home_id) ? 
				globalDataWindow[home_of_rdata] : 
				node_alter_tbl[home_of_rdata].alter_globalDataWindow);
	home_of_rdata = node_alter_tbl[home_of_rdata].alter_home_id;
	
	/* CSP: Get the lock */
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, repl_node, 0, 
			node_alter_tbl_window[dead_node]);
	MPI_Compare_and_swap(&original_bool, &compare_bool, &result_bool, MPI_C_BOOL,
			repl_node, recovering_offset, node_alter_tbl_window[dead_node]);
	MPI_Win_unlock(repl_node, node_alter_tbl_window[dead_node]);

	if (!result_bool) {
		/* CSP: other nodes may have finished writing before this node
		 *  even starts to try and get the lock. Doublecheck this first.
		 * */
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, argo_get_nid(), 0, 
				node_alter_tbl_window[dead_node]);
		MPI_Get(&finished_work, 1, MPI_C_BOOL, argo_get_nid(), 
				just_recovered_offset,	1, MPI_C_BOOL, node_alter_tbl_window[dead_node]);
		MPI_Win_unlock(argo_get_nid(), node_alter_tbl_window[dead_node]);

		if (!finished_work) {
			/* CSP: Rebuild node data */
			/* CSP TODO:
			 * For complete replication nothing is needed for now. For EC, replace the
			 * 	repl_data on the repl_node with the lost original data.
			 * */

			if (env::replication_recovery_policy() == 1) {
				/* CSP: Rebuild data as alternative node */
		
				/* CSP: Prepare an alt_tbl to write. */
				temp_tbl.alter_home_id = argo_get_nid();
				temp_tbl.alter_globalData 
						= static_cast<char*>(vm::allocate_mappable(pagesize, size_of_chunk));
				temp_tbl.just_recovered = true;
				temp_tbl.alter_repl_id = argo_get_nid();
				temp_tbl.alter_replData 
						= static_cast<char*>(vm::allocate_mappable(pagesize, size_of_chunk));

				/* CSP: map data area */
				vm::map_memory((void *)(temp_tbl.alter_globalData), size_of_chunk, 
								vm_map_offset_record, PROT_READ|PROT_WRITE);
				vm_map_offset_record += size_of_chunk;
				vm::map_memory((void *)(temp_tbl.alter_globalData), size_of_chunk, 
								vm_map_offset_record, PROT_READ|PROT_WRITE);
				vm_map_offset_record += size_of_chunk;

				/* CSP: copy and rebuild data; local node is alternative node */
				sem_wait(&ibsem);
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, repl_node, 0, rdata_source_win);
				MPI_Get(temp_tbl.alter_globalData, size_of_chunk, MPI_BYTE, repl_node, 
						0, size_of_chunk, MPI_BYTE, rdata_source_win);
				MPI_Win_unlock(repl_node, rdata_source_win);
				sem_post(&ibsem);

				sem_wait(&ibsem);
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, home_of_rdata, 0, gdata_source_win);
				MPI_Get(temp_tbl.alter_replData, size_of_chunk, MPI_BYTE, home_of_rdata, 
						0, size_of_chunk, MPI_BYTE, gdata_source_win);
				MPI_Win_unlock(home_of_rdata, gdata_source_win);
				sem_post(&ibsem);

				/* CSP: Update table locally and attatch memory. */
				/* CSP TODO: Do this with MPI? Will there be race conditions? */
				MPI_Win_attach(node_alter_tbl[dead_node].alter_globalDataWindow, 
						temp_tbl.alter_globalData, size_of_chunk*sizeof(argo_byte));
				MPI_Win_attach(node_alter_tbl[dead_node].alter_replDataWindow, 
						temp_tbl.alter_replData, size_of_chunk*sizeof(argo_byte));
				node_alter_tbl[dead_node].alter_home_id = temp_tbl.alter_home_id;
				node_alter_tbl[dead_node].alter_globalData = temp_tbl.alter_globalData;
				node_alter_tbl[dead_node].alter_repl_id = temp_tbl.alter_repl_id;
				node_alter_tbl[dead_node].alter_replData = temp_tbl.alter_replData;

				/* CSP: Update all tables on all nodes. Note: update id=dead_node. */
				for (int i = 0; i < numtasks; ++i) {
					if (i != dead_node && i != argo_get_nid()) {
						MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i, 0, node_alter_tbl_window[dead_node]);
					}
				}
				for (int i = 0; i < numtasks; ++i) {
					if (i != dead_node && i != argo_get_nid()) {
						// CSP: Skip MPI windows and bool recovering
						MPI_Put(&(temp_tbl.alter_home_id), sizeof(argo::node_id_t), MPI_BYTE, i, 
								alter_hid_offset, sizeof(argo::node_id_t), 
								MPI_BYTE, node_alter_tbl_window[dead_node]);
						MPI_Put(&(temp_tbl.alter_globalData), sizeof(char*), MPI_BYTE, i, 
								alter_gdata_offset, sizeof(char*), 
								MPI_BYTE, node_alter_tbl_window[dead_node]);
						MPI_Put(&(temp_tbl.just_recovered), sizeof(char*), MPI_BYTE, i, 
								just_recovered_offset, sizeof(char*), 
								MPI_BYTE, node_alter_tbl_window[dead_node]);
						MPI_Put(&(temp_tbl.alter_repl_id), sizeof(argo::node_id_t), MPI_BYTE, i, 
								alter_rid_offset, sizeof(argo::node_id_t), 
								MPI_BYTE, node_alter_tbl_window[dead_node]);
						MPI_Put(&(temp_tbl.alter_replData), sizeof(char*), MPI_BYTE, i, 
								alter_rdata_offset, sizeof(char*), 
								MPI_BYTE, node_alter_tbl_window[dead_node]);
					}
				} 
				for (int i = 0; i < numtasks; ++i) {
					// CSP: Unlock all nodes together to "commit" all changes altogether
					if (i != dead_node && i != argo_get_nid()) {
						MPI_Win_unlock(i, node_alter_tbl_window[dead_node]);
					}
				}
			} else if (env::replication_recovery_policy() == 0) {
				/* CSP: Redirect all data access to replication node */

				if (env::replication_policy() == 1) { // complete replication
					temp_tbl.alter_home_id = repl_node;	// write to repl node
					temp_tbl.just_recovered = true; // update MPI window locally
					temp_tbl.alter_repl_id = -1; // skip writing repl data

					// Update locally
					node_alter_tbl[dead_node].alter_home_id = repl_node;
					node_alter_tbl[dead_node].just_recovered = true;
					node_alter_tbl[dead_node].alter_repl_id = -1;
				} else if (env::replication_policy() == 2) { // EC
					// Need to decide the destination dynamically.
					temp_tbl.alter_home_id = -1; // a special marker
					temp_tbl.just_recovered = true; // update MPI window locally
					temp_tbl.alter_repl_id = -1; // skip writing repl data

					// Update locally
					node_alter_tbl[dead_node].alter_home_id = -1;
					node_alter_tbl[dead_node].just_recovered = true;
					node_alter_tbl[dead_node].alter_repl_id = -1;
				}

				if (env::replication_policy() == 2) { //EC
					/* CSP TODO:
					 *
					 * How to prevent other nodes from writing to the repl pages?
					 *
				 	 * We should first update the table, and then recover data.
					 * The updated table should have a flag that tells all nodes to 
				 	 *  restrain from writing to the repl pages that is used in the 
				 	 *  recovering procedure.
				 	 * */

					char page_buffer[pagesize];
					std::size_t page_addr = 0; // page on dead_node
					std::size_t check_addr = 0;	// page to check (if it has the same repl page)
					std::size_t addr_max = argo::virtual_memory::size();
					std::size_t page_repl_offset, check_offset;
					argo::node_id_t page_repl_id, check_home_id;
					/* CSP: look through all pages to find those on dead_node
					 * TODO: this looks at every page, wherever they are and 
					 *  whether it is been utilized. Any optimizations?
					 * */
					for (page_addr = 0; page_addr < addr_max; page_addr += pagesize) {
						if (peek_homenode(page_addr) != dead_node) {
							continue; // skip if page_addr not on dead_node
						}

						page_repl_id = get_replication_node(page_addr); // node id of repl page
						page_repl_offset = get_replication_offset(page_addr); // repl page offset on that node
						
						// For each page on dead_node, look for all pages writing to the same repl page
						for (check_addr = 0; check_addr < addr_max; check_addr += pagesize) {
							check_home_id = peek_homenode(check_addr);
							check_offset = peek_offset(check_addr);
							if (check_addr != page_addr
									&& get_replication_node(check_addr) == page_repl_id
									&& get_replication_offset(check_addr) == page_repl_offset) {
								// This page (starting from check_addr) writes to the same repl page
								MPI_Win_lock(MPI_LOCK_EXCLUSIVE, check_home_id, 0, replDataWindow[check_home_id]);
								MPI_Get(page_buffer, pagesize, MPI_BYTE, 
										check_home_id, check_offset, pagesize, 
										MPI_BYTE, replDataWindow[check_home_id]);
								MPI_Win_unlock(check_home_id, replDataWindow[check_home_id]);
	
								MPI_Win_lock(MPI_LOCK_EXCLUSIVE, page_repl_id, 0, replDataWindow[page_repl_id]);
								MPI_Accumulate(page_buffer, page_size, MPI_BYTE, 
										page_repl_id, page_repl_offset, page_size, 
										MPI_BYTE, MPI_BXOR, replDataWindow[page_repl_id]);
								MPI_Win_unlock(page_repl_id, replDataWindow[page_repl_id]);
							}
						}
					} // for (page_addr = 0; page_addr < addr_max)
				} // if (env::replication_policy == 2)

				node_alter_tbl[dead_node].alter_globalData = temp_tbl.alter_globalData;
				node_alter_tbl[dead_node].alter_repl_id = temp_tbl.alter_repl_id;
				node_alter_tbl[dead_node].alter_replData = temp_tbl.alter_replData;

				/* CSP: Update all tables on all nodes. Note: update id=dead_node. */
				for (int i = 0; i < numtasks; ++i) {
					if (i != dead_node && i != argo_get_nid()) {
						MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i, 0, node_alter_tbl_window[dead_node]);
					}
				}
				for (int i = 0; i < numtasks; ++i) {
					if (i != dead_node && i != argo_get_nid()) {
						// CSP: Skip MPI windows and bool recovering
						MPI_Put(&(temp_tbl.alter_home_id), 
								sizeof(argo::node_id_t), MPI_BYTE, i, 
								alter_hid_offset, sizeof(argo::node_id_t), 
								MPI_BYTE, node_alter_tbl_window[dead_node]);
						MPI_Put(&(temp_tbl.just_recovered),
								sizeof(char*), MPI_BYTE, i, 
								just_recovered_offset, sizeof(char*), 
								MPI_BYTE, node_alter_tbl_window[dead_node]);
						MPI_Put(&(temp_tbl.alter_repl_id), 
								sizeof(argo::node_id_t), MPI_BYTE, i, 
								alter_rid_offset, sizeof(argo::node_id_t), 
								MPI_BYTE, node_alter_tbl_window[dead_node]);
					}
				}
				for (int i = 0; i < numtasks; ++i) {
					// CSP: Unlock all nodes together to "commit" all changes altogether
					if (i != dead_node && i != argo_get_nid()) {
						MPI_Win_unlock(i, node_alter_tbl_window[dead_node]);
					}
				}
			} // if (env::replication_recovery_policy() == ...)
		} // if (!finished_work)
		/* CSP: Release the lock */
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, repl_node, 0, node_alter_tbl_window[dead_node]);
		MPI_Put(&compare_bool, 1, MPI_C_BOOL, repl_node, recovering_offset, 1, MPI_C_BOOL, node_alter_tbl_window[dead_node]);
		MPI_Win_unlock(repl_node, node_alter_tbl_window[dead_node]);
	} else {
		/* CSP: Use MPI to access local mem because it may be updated by MPI */
		fprintf(stderr, "%d: ----before: hid: %d, gData: %p, gwindow: %p, grefresh: %d, rid: %d, rdata: %p, rwindow: %p, rrefresh: %d, rebuild: %d\n", 
					argo_get_nid(),
					(node_alter_tbl[dead_node].alter_home_id),
					(node_alter_tbl[dead_node].alter_globalData),
					(node_alter_tbl[dead_node].alter_globalDataWindow),
					(node_alter_tbl[dead_node].just_recovered),
					(node_alter_tbl[dead_node].alter_repl_id),
					(node_alter_tbl[dead_node].alter_replData),
					(node_alter_tbl[dead_node].alter_replDataWindow),
					(node_alter_tbl[dead_node].refresh_replDataWindow),
					(node_alter_tbl[dead_node].recovering)
		);
		while (!finished_work) {
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, argo_get_nid(), 0, node_alter_tbl_window[dead_node]);
			MPI_Get(&finished_work, 1, MPI_C_BOOL, argo_get_nid(),
				just_recovered_offset,	1, MPI_C_BOOL, node_alter_tbl_window[dead_node]);
			MPI_Win_unlock(argo_get_nid(), node_alter_tbl_window[dead_node]);
		}
		fprintf(stderr, "%d: ----after: hid: %d, gData: %p, gwindow: %p, grefresh: %d, rid: %d, rdata: %p, rwindow: %p, rrefresh: %d, recovering: %d\n", 
					argo_get_nid(),
					(node_alter_tbl[dead_node].alter_home_id),
					(node_alter_tbl[dead_node].alter_globalData),
					(node_alter_tbl[dead_node].alter_globalDataWindow),
					(node_alter_tbl[dead_node].just_recovered),
					(node_alter_tbl[dead_node].alter_repl_id),
					(node_alter_tbl[dead_node].alter_replData),
					(node_alter_tbl[dead_node].alter_replDataWindow),
					(node_alter_tbl[dead_node].refresh_replDataWindow),
					(node_alter_tbl[dead_node].recovering)
		);
	}
}

/* CSPext: Exposed rebuild function for testing purpose */
/**
 * @brief Calls redundancy_rebuild. Exposed in argo::backend namespace.
 * @param dead_node Id of the node which is down.
 * */
void argo_test_interface_rebuild(argo::node_id_t dead_node) {
	lost_node_data_recovery(dead_node);
}

/* CSPext: Create or re-create replDataWindow */
void local_check_after_recovery(node_alternation_table *tbl) {
	if (tbl->just_recovered) {
		if (env::replication_policy() == 2 
						&& env::replication_recovery_policy() == 0) {
			// EC + rebuilding: update no_repl_page
			std::size_t page_addr = 0; // page on dead_node
			std::size_t check_addr = 0;	// page to check (if it has the same repl page)
			std::size_t addr_max = argo::virtual_memory::size();
			std::size_t page_repl_offset;
			argo::node_id_t page_repl_id, check_home_id;
			/* CSP: look through all pages to find those on this node
			 * TODO: this looks at every page, wherever they are and 
			 *  whether it is been utilized. Any optimizations?
			 * */
			for (page_addr = 0; page_addr < addr_max; page_addr += pagesize) {
				if (peek_homenode(page_addr) != argo_get_nid()) {
					continue; // skip if page_addr not on this node
				}

				page_repl_id = get_replication_node(page_addr); // node id of repl page
				page_repl_offset = get_replication_offset(page_addr); // repl page offset on that node
				if (no_repl_page[page_repl_offset / size_of_all]) {
					// already recorded
					continue;
				}
						
				// For each page on current node, 
				//  check if any of its repl page is utilized as alternative
				for (check_addr = 0; check_addr < addr_max; check_addr += pagesize) {
					check_home_id = peek_homenode(check_addr);
					if (check_addr == page_addr 
							|| node_alter_tbl[check_home_id].alter_home_id != -1) {
						// node is till running
						continue;
					}

					if (get_replication_node(check_addr) == page_repl_id
							&& get_replication_offset(check_addr) == page_repl_offset) {
						// This page (starting from check_addr) writes to the same repl page
						no_repl_page[page_repl_offset / size_of_all] = true;
					}
				}
			} // for (page_addr = 0; page_addr < addr_max)
		}
		if (env::replication_recovery_policy() == 0) {
			// No need to do this to repl data
			tbl->alter_globalDataWindow = replDataWindow[tbl->alter_home_id];
		}
		tbl->just_recovered = false;
	}
}

/* CSPext: Create or re-create replDataWindow */
void node_alter_tbl_rdata_local_response(node_alternation_table *tbl) {
	if (tbl->refresh_replDataWindow) {
		tbl->refresh_replDataWindow = false;
	}
}

/* CSPext: A function to copy data from the input pointer's repl node */
void get_replicated_data(dd::global_ptr<char> ptr, void* container, unsigned int len) {
	/*
	 * CSP TODO:
	 * Update table-checking
	 * */
	const argo::node_id_t h = ptr.peek_node();
	const argo::node_id_t r = ptr.get_replication_node();	// repl node id
	const std::size_t offset = ptr.get_replication_offset();

	// printf("----get repl data: Node %d: array[0] = %d, container[0] = %d\n", getID(), ((int *) (replData + ptr.offset()))[0], ((int *) container)[0]);
	// printf("----get repl data: ptr %p container %p\n", ptr.get(), container);
	// printf("----get repl data: h %d r %d offset %lu length %u\n", h, r, offset, len);
	
	if (h == dd::invalid_node_id || r == dd::invalid_node_id) {
		// TODO: Do nothing and return. Or what should we do?
		return;
	}
	if (getID() == r) {
		memcpy(container, replData + offset, len);
		return;
	} else {
		// Lock needed? (Probably)
		//printf("Start address: %p; ptr address: %p; ptr offset: %lu; globalData: %p\n", 
		//				argo::backend::global_base(), ptr.get(), ptr.offset(), globalData);

		//printf("ptr.get(): %x, globalData + offset(): %x\n", 
		//				*ptr.get(), *(globalData + ptr.offset()));
		sem_wait(&ibsem);
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, r, 0, replDataWindow[r]);
		MPI_Get(container, len, MPI_BYTE, r, offset, len, MPI_BYTE, replDataWindow[r]);
		MPI_Win_unlock(r, replDataWindow[r]);
		sem_post(&ibsem);
	}
}

