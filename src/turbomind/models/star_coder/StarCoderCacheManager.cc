// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/star_coder/StarCoderCacheManager.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

StarCoderCacheManager::~StarCoderCacheManager()
{
    for (auto& p : device_mem_) {
        allocator_->free(&p, false);
    }
}

void* StarCoderCacheManager::allocate(bool is_preallocte)
{
    if (rank_ == 0) {
        TM_LOG_INFO("[StarCoderCacheManager][allocate]");
    }

    void* mem_ptr{};

    if (!device_free_.empty()) {
        mem_ptr = device_free_.front();
        device_free_.pop();

        if (rank_ == 0) {
            TM_LOG_INFO("[StarCoderCacheManager][allocate] free = %d", (int)device_free_.size());
        }
    }
    else if (entry_count_ < max_entry_count_) {
        const auto   alloc_count     = std::min(chunk_size_, max_entry_count_ - entry_count_);
        const size_t entry_byte_size = 2 * cache_byte_size_;  // 2 for k,v

        if (rank_ == 0) {
            TM_LOG_INFO("[StarCoderCacheManager][allocate] malloc %d", (int)alloc_count);
        }
        const auto chunk_ptr = allocator_->malloc(alloc_count * entry_byte_size, false);
        FT_CHECK(chunk_ptr);
        device_mem_.push_back(chunk_ptr);
        entry_count_ += alloc_count;
        if (rank_ == 0) {
            TM_LOG_INFO("[StarCoderCacheManager][allocate] count = %d", entry_count_);
        }

        for (int i = 0; i < alloc_count; ++i) {
            device_free_.push((uint8_t*)chunk_ptr + entry_byte_size * i);
        }

        if (!is_preallocte) {
            mem_ptr = device_free_.front();
            device_free_.pop();
        }

        if (rank_ == 0) {
            TM_LOG_INFO("[StarCoderCacheManager][allocate] free = %d", (int)device_free_.size());
        }
    }
    else {
        mem_ptr = evict();
        FT_CHECK_WITH_INFO(mem_ptr, "No enough cache entries.");
    }

    return mem_ptr;
}

auto StarCoderCacheManager::create(uint64_t id, cudaStream_t stream) -> Sequence
{
    if (rank_ == 0) {
        TM_LOG_INFO("[StarCoderCacheManager][create] %ld", (long)id);
    }

    for (const auto& e : device_cache_) {
        if (e.id == id) {
            if (rank_ == 0) {
                TM_LOG_WARNING("[StarCoderCacheManager][create] Removing conflicting id %ld", (long)id);
            }
            erase(id);
        }
    }

    const auto mem_ptr = (uint8_t*)allocate(false);
    check_cuda_error(cudaMemsetAsync(mem_ptr, 0, cache_byte_size_ * 2, stream));

    device_cache_.push_back({
        id,
        max_seq_len_,
        {},
        0,
        mem_ptr,
        mem_ptr + cache_byte_size_,
        {},
        static_cast<uint64_t>(-1),
    });

    return device_cache_.back();
}

auto StarCoderCacheManager::getEntryOrThrow(uint64_t id) -> std::vector<Sequence>::iterator
{
    auto pred = [&](const Sequence& s) { return s.id == id; };
    auto it   = std::find_if(device_cache_.begin(), device_cache_.end(), pred);
    if (it == device_cache_.end()) {
        TM_LOG_ERROR("[StarCoderCacheManager] %ld not found.\n", (long)id);
        FT_CHECK(0);
    }
    return it;
}

auto StarCoderCacheManager::fetch(uint64_t id, cudaStream_t stream) -> Sequence
{
    if (rank_ == 0) {
        TM_LOG_INFO("[StarCoderCacheManager][fetch] %ld", (long)id);
    }

    auto entry = getEntryOrThrow(id);

    if (entry->k_cache == nullptr) {
        FT_CHECK(entry->cache_len == 0);
        const auto mem_ptr = allocate(false);
        check_cuda_error(cudaMemsetAsync(mem_ptr, 0, cache_byte_size_ * 2, stream));
        entry->k_cache = mem_ptr;
        entry->v_cache = (uint8_t*)entry->k_cache + cache_byte_size_;
    }

    entry->timestamp = static_cast<uint64_t>(-1);
    return *entry;
}

void StarCoderCacheManager::update(const Sequence& seq, cudaStream_t stream)
{
    if (rank_ == 0) {
        TM_LOG_INFO("[StarCoderCacheManager][update] %ld", (long)seq.id);
    }

    auto entry = getEntryOrThrow(seq.id);

    entry->timestamp = ++timestamp_;
    entry->token_ids = seq.token_ids;
    entry->cache_len = seq.cache_len;
    FT_CHECK(seq.k_cache == entry->k_cache && seq.v_cache == entry->v_cache);
}

void StarCoderCacheManager::erase(uint64_t id)
{
    if (rank_ == 0) {
        TM_LOG_INFO("[StarCoderCacheManager][erase] %ld", (long)id);
    }

    auto entry = getEntryOrThrow(id);

    if (entry->k_cache) {
        device_free_.push(entry->k_cache);
        if (rank_ == 0) {
            TM_LOG_INFO("[StarCoderCacheManager][erase] free = %d", (int)device_free_.size());
        }
    }
    device_cache_.erase(entry);
}

void* StarCoderCacheManager::evict()
{
    FT_CHECK(!device_cache_.empty());
    auto it = std::min_element(device_cache_.begin(), device_cache_.end(), [](const auto& a, const auto& b) {
        return a.timestamp < b.timestamp;
    });

    if (it->timestamp == static_cast<uint64_t>(-1)) {
        return nullptr;
    }

    if (rank_ == 0) {
        TM_LOG_INFO("[StarCoderCacheManager][evict] %ld", (long)it->id);
    }

    FT_CHECK(it->k_cache);
    auto mem_ptr = it->k_cache;
    it->k_cache = it->v_cache = nullptr;
    it->cache_len             = 0;
    it->timestamp             = static_cast<uint64_t>(-1);
    return mem_ptr;
}

bool StarCoderCacheManager::contains(uint64_t id) const noexcept
{
    auto pred = [&](const Sequence& s) { return s.id == id; };
    auto it   = std::find_if(device_cache_.begin(), device_cache_.end(), pred);
    return it != device_cache_.end();
}

}  // namespace turbomind
