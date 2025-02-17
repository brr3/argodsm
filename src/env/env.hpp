/**
 * @file
 * @brief This file provides facilities for handling environment variables
 * @copyright Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.
 */

#ifndef argo_env_env_hpp
#define argo_env_env_hpp argo_env_env_hpp

#include <cstddef>

/**
 * @page envvars Environment Variables
 *
 * @envvar{ARGO_MEMORY_SIZE} request a specific memory size in bytes
 * @details This environment variable is only used if argo::init() is called with no
 *          argo_size parameter (or it has value 0). It can be accessed through
 *          @ref argo::env::memory_size() after argo::env::init() has been called.
 *
 * @envvar{ARGO_CACHE_SIZE} request a specific cache size in bytes
 * @details This environment variable is only used if argo::init() is called with no
 *          cache_size parameter (or it has value 0). It can be accessed through
 *          @ref argo::env::cache_size() after argo::env::init() has been called.
 *
 * @envvar{ARGO_WRITE_BUFFER_SIZE} request a specific write buffer size in cache blocks
 * @details This environment variable defaults to 512 cache blocks if not specified.
 *          It can be accessed through @ref argo::env::write_buffer_size() after
 *          argo::env::init() has been called.
 *
 * @envvar{ARGO_WRITE_BUFFER_WRITE_BACK_SIZE} request a specific write buffer write
 * back size in cache blocks.
 * @details This environment variable defaults to 32 cache blocks if not specified.
 *          It can be accessed through @ref argo::env::write_buffer_write_back_size()
 *          after argo::env::init() has been called.
 * 
 * @envvar{ARGO_ALLOCATION_POLICY} request a specific allocation policy with a number
 * @details This environment variable can be accessed through
 *          @ref argo::env::allocation_policy() after argo::env::init() has been called.
 * 
 * @envvar{ARGO_REPLICATION_POLICY} request a specific replication policy with a number
 * @details This environment variable can be accessed through
 *          @ref argo::env::replication_policy() after argo::env::init() has been called.
 * 
 * @envvar{ARGO_ALLOCATION_BLOCK_SIZE} request a specific allocation block size in number of pages
 * @details This environment variable can be accessed through
 *          @ref argo::env::allocation_block_size() after argo::env::init() has been called.
 *
 * @envvar{ARGO_LOAD_SIZE} request a specific load size in number of pages
 * @details This environment variable determines the maximum amount of DSM pages that
 * 			are fetched on each remote load operation. It can be accessed through
 *          @ref argo::env::load_size() after argo::env::init() has been called.
 */

namespace argo {
	/**
	 * @brief namespace for environment variable handling functionality
	 * @note argo::env::init() must be called before other functions in this
	 *       namespace work correctly
	 */
	namespace env {
		/**
		 * @brief read and store environment variables used by ArgoDSM
		 * @details the environment is only read once, to avoid having
		 *          to check that values are not changing later
		 */
		void init();

		/**
		 * @brief get the memory size requested by environment variable
		 * @return the requested memory size in bytes
		 * @see @ref ARGO_MEMORY_SIZE
		 */
		std::size_t memory_size();

		/**
		 * @brief get the cache size requested by environment variable
		 * @return the requested cache size in bytes
		 * @see @ref ARGO_CACHE_SIZE
		 */
		std::size_t cache_size();

		/**
		 * @brief get the write buffer size requested by environment variable
		 * @return the requested write buffer size in cache blocks
		 * @see @ref ARGO_WRITE_BUFFER_SIZE
		 */
		std::size_t write_buffer_size();

		/**
		 * @brief get the write buffer write back size requested by environment variable
		 * @return the requested write buffer write back size in cache blocks
		 * @see @ref ARGO_WRITE_BUFFER_WRITE_BACK_SIZE
		 */
		std::size_t write_buffer_write_back_size();
		
		/**
		 * @brief get the allocation policy requested by environment variable
		 * @return the requested allocation policy as a number
		 * @see @ref ARGO_ALLOCATION_POLICY
		 */
		std::size_t allocation_policy();

		/**
		 * @brief get the allocation block size requested by environment variable
		 * @return the requested allocation block size as a number of pages
		 * @see @ref ARGO_ALLOCATION_BLOCK_SIZE
		 */
		std::size_t allocation_block_size();

		// CSPext

		/**
		 * @brief get the replication policy requested by environment variable
		 * @return the requested replication policy as a number:
		 * 0 == no replication, 1 == complete replication, 2 == erasure code (n-1, 1)
		 */
		std::size_t replication_policy();

		/**
		 * @brief get the replication data fragments requested by environment variable
		 * @return the requested replication data fragments as a number
		 */
		std::size_t replication_data_fragments();

		/**
		 * @brief get the replication parity fragments requested by environment variable
		 * @return the requested replication parity fragments as a number
		 */
		std::size_t replication_parity_fragments();

		/**
		 * @brief get the replication recovery policy requested by environment variable
		 * @return the requested replication recovery policy as a number
		 */
		std::size_t replication_recovery_policy();
		// CSPext end

		/**
		 * @brief get the number of pages fetched remotely per load requested
		 * by environment variable
		 * @return the number of pages
		 * @see @ref ARGO_LOAD_SIZE
		 */
		std::size_t load_size();
	} // namespace env
} // namespace argo

#endif
