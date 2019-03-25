#include "utils/thread_pool.h"

namespace utils {

ThreadPool::ThreadPool(size_t nr_threads)
{
    extend(nr_threads);
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard lock{tasks_mutex_};
        stopping_ = true;
    }

    tasks_condvar_.notify_all();
    for (std::thread& worker : workers_)
        worker.join();
}

size_t ThreadPool::size() const
{
    return workers_.size();
}

void ThreadPool::extend(size_t nr_threads)
{
    workers_.reserve(workers_.size() + nr_threads);
    for (size_t i = 0; i < nr_threads; ++i)
        workers_.emplace_back([this] {
            while (true) {
                std::unique_lock lock{tasks_mutex_};
                tasks_condvar_.wait(lock, [&] {
                    return stopping_ || !tasks_.empty();
                });
                if (stopping_ && tasks_.empty())
                    return;

                std::function<void()> task = std::move(tasks_.front());
                tasks_.pop();
                lock.unlock();

                task();
            }
        });
}

}  // namespace utils
