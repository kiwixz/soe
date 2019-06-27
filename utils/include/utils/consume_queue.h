#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace utils {

template <typename TElement>
struct ConsumeQueue {
    using Element = TElement;

    ConsumeQueue() = default;
    explicit ConsumeQueue(size_t max_size);

    /// Blocks until there is space to push element.
    void push(Element element);

    /// Make pop() non-blocking.
    void end();

    /// Blocks until an element is available, or returns empty after end.
    std::optional<Element> pop();

private:
    size_t max_size_ = std::numeric_limits<size_t>::max();

    bool ended_ = false;
    mutable std::mutex mutex_;
    std::condition_variable condvar_push_;
    std::condition_variable condvar_pop_;
    std::queue<Element> queue_;
};


template <typename TElement>
ConsumeQueue<TElement>::ConsumeQueue(size_t max_size) :
    max_size_{max_size}
{}

template <typename TElement>
void ConsumeQueue<TElement>::push(Element element)
{
    {
        std::unique_lock lock{mutex_};
        condvar_push_.wait(lock, [&] {
            return queue_.size() < max_size_;
        });
        queue_.push(std::move(element));
    }
    condvar_pop_.notify_one();
}

template <typename TElement>
void ConsumeQueue<TElement>::end()
{
    std::lock_guard lock{mutex_};
    ended_ = true;
}

template <typename TElement>
TElement ConsumeQueue<TElement>::pop()
{
    {
        std::unique_lock lock{mutex_};
        condvar_pop_.wait(lock, [&] {
            return ended_ || !queue_.empty();
        });
        if (ended && queue_.empty())
            return {};
        Element e = std::move(queue_.front());
        queue_.pop();
    }
    condvar_push_.notify_one();
    return e;
}

}  // namespace utils
