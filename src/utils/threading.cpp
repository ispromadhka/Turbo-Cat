/**
 * TurboCat Threading Utilities
 * 
 * Thread pool and parallel execution helpers.
 */

#include "turbocat/types.hpp"
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace turbocat {
namespace threading {

// ============================================================================
// Thread Pool (for non-OpenMP builds)
// ============================================================================

class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads = 0) {
        if (n_threads == 0) {
            n_threads = std::thread::hardware_concurrency();
        }
        
        for (size_t i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });
                        
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        
        for (auto& worker : workers_) {
            worker.join();
        }
    }
    
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        
        using return_type = typename std::invoke_result<F, Args...>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        
        condition_.notify_one();
        return result;
    }
    
    size_t n_threads() const {
        return workers_.size();
    }
    
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stop_ = false;
};

// Global thread pool
static ThreadPool* global_pool = nullptr;
static std::mutex pool_mutex;

ThreadPool& get_global_pool() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (!global_pool) {
        global_pool = new ThreadPool();
    }
    return *global_pool;
}

void shutdown_global_pool() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (global_pool) {
        delete global_pool;
        global_pool = nullptr;
    }
}

// ============================================================================
// Parallel For
// ============================================================================

void parallel_for(size_t begin, size_t end, std::function<void(size_t)> body) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = begin; i < end; ++i) {
        body(i);
    }
    #else
    // Use thread pool
    auto& pool = get_global_pool();
    std::vector<std::future<void>> futures;
    
    for (size_t i = begin; i < end; ++i) {
        futures.push_back(pool.submit([&body, i]() { body(i); }));
    }
    
    for (auto& f : futures) {
        f.wait();
    }
    #endif
}

void parallel_for_blocked(size_t begin, size_t end, size_t block_size,
                          std::function<void(size_t, size_t)> body) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t block_start = begin; block_start < end; block_start += block_size) {
        size_t block_end = std::min(block_start + block_size, end);
        body(block_start, block_end);
    }
    #else
    auto& pool = get_global_pool();
    std::vector<std::future<void>> futures;
    
    for (size_t block_start = begin; block_start < end; block_start += block_size) {
        size_t block_end = std::min(block_start + block_size, end);
        futures.push_back(pool.submit([&body, block_start, block_end]() {
            body(block_start, block_end);
        }));
    }
    
    for (auto& f : futures) {
        f.wait();
    }
    #endif
}

// ============================================================================
// Thread-Local Storage
// ============================================================================

int get_thread_id() {
    #ifdef _OPENMP
    return omp_get_thread_num();
    #else
    static thread_local int id = -1;
    static std::atomic<int> counter{0};
    if (id < 0) {
        id = counter.fetch_add(1);
    }
    return id;
    #endif
}

int get_max_threads() {
    #ifdef _OPENMP
    return omp_get_max_threads();
    #else
    return static_cast<int>(std::thread::hardware_concurrency());
    #endif
}

void set_num_threads(int n) {
    #ifdef _OPENMP
    omp_set_num_threads(n);
    #else
    // Recreate thread pool with new size
    shutdown_global_pool();
    global_pool = new ThreadPool(n > 0 ? n : std::thread::hardware_concurrency());
    #endif
}

} // namespace threading
} // namespace turbocat
