#include "ThreadPool.h"
#include <thread>
#include <assert.h>
#include <iostream>
#include "Logging.h"

ThreadPool::ThreadPool(int nthreads) : _nthreads(nthreads)
{
	LOG_DEBUG("ThreadPool: Starting {} threads", nthreads);
	//spin up the actual threads
	for (int i = 0; i < _nthreads; i++) {
		_threads.push_back(std::thread(&ThreadPool::SchedulerLoop, this,  i));
	}
}

ThreadPool::~ThreadPool()
{	
	_threads.clear();
}

void ThreadPool::SchedulerLoop(int thread_index)
{
	while (true) {
		Task* job = nullptr;
		{
			std::unique_lock jobs_lock(_scheduled_jobs_mutex);
			if (!_scheduled.empty()) {
				job =_scheduled.front();
				_scheduled.pop();
			}
		}
		
		if (job == nullptr) {
			//std::this_thread::yield();
			continue;
		}
		else {
			LOG_TRACE("starting job {}, {} on thread {}", job->data_index, job->data_len, thread_index);
			job->set_thread_index(thread_index);
			job->task(thread_index, job->data_index, job->data_len);
			job->mark_complete();
			LOG_TRACE("job complete on thread {}", thread_index);
		}
	}
}

void ThreadPool::schedule(Task *task, int on_thread)
{
	std::unique_lock jobs_lock(_scheduled_jobs_mutex);
	_scheduled.push(task);
}


void ThreadPool::batch_jobs(batch_function& func, size_t data_len)
{
	size_t batch_size = data_len /_nthreads + 1;
	size_t data_index = 0;
	std::vector<std::unique_ptr<Task>> tasks;
	for (int i = 0; i < _nthreads; i++) {
		
		size_t len = std::min(batch_size, data_len - data_index);
		auto task = std::make_unique<Task>(func, data_index, len);
		schedule(task.get(), i);

		tasks.push_back(std::move(task));
		LOG_TRACE("submitted job {} {}", data_index, len);
		data_index += len;
	}
	assert(data_index == data_len);

	for (auto &task : tasks) {
		LOG_TRACE("waiting on i={}", task->get_thread_index());
		while (!task->is_complete()) {
			std::this_thread::yield();
		}
		LOG_TRACE("i={} complete", task->get_thread_index());
	}
}
