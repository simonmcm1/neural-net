#include "ThreadPool.h"
#include <thread>
#include <assert.h>
#include <iostream>
#include "Logging.h"

void Task::wait_for_complete() {
	std::unique_lock lock(is_complete_mutex);
	is_complete_cv.wait(lock, [&] {return _is_complete;});
}

void Task::mark_complete() {
	std::unique_lock lock(is_complete_mutex);
	_is_complete = true;
	lock.unlock();
	is_complete_cv.notify_all();
}

ThreadPool::ThreadPool(int nthreads) : _nthreads(nthreads)
{
	LOG_DEBUG("ThreadPool: Starting {} threads", nthreads);
	//spin up the actual threads
	for (int i = 0; i < _nthreads; i++) {
		_threads.push_back(std::thread(&ThreadPool::SchedulerLoop, this,  i));
	}

	_scheduled_fixed_thread.resize(nthreads);
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
			std::scoped_lock jobs_lock(_scheduled_jobs_mutex);
			auto& this_thread_queue = _scheduled_fixed_thread.at(thread_index);
			if (!this_thread_queue.empty()) {
				job = this_thread_queue.front();
				this_thread_queue.pop();
			} else if (!_scheduled.empty()) {
				job =_scheduled.front();
				_scheduled.pop();
			}
		}
		
		if (job == nullptr) {
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
	std::scoped_lock jobs_lock(_scheduled_jobs_mutex);
	if (on_thread < 0) {
		_scheduled.push(task);
	}
	else {
		_scheduled_fixed_thread.at(on_thread).push(task);
	}
	
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
		task->wait_for_complete();
		LOG_TRACE("i={} complete", task->get_thread_index());
	}
}
