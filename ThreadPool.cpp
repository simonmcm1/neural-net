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

//ThreadPool ThreadPool::global_instance(std::thread::hardware_concurrency() - 2);

ThreadPool::ThreadPool(int nthreads) : _nthreads(nthreads), _workers(nthreads)
{
	LOG_DEBUG("ThreadPool: Starting {} threads", nthreads);

	//spin up the actual threads
	for (int i = 0; i < _nthreads; i++) {
		_workers.at(i).thread = std::thread(&ThreadPool::SchedulerLoop, this, i);
	}

	
}

ThreadPool::~ThreadPool()
{	
	for (auto& t : _workers) {
		t.terminate = true;
	}
	for (auto& t : _workers) {
		t.thread.join();
	}
	_workers.clear();
}

void ThreadPool::SchedulerLoop(int thread_index)
{
	auto& this_worker = _workers.at(thread_index);
	while (true) {
		Task* job = nullptr;
		{
			if (terminate) {
				break;
			}

			std::unique_lock jobs_lock(this_worker.scheduled_mutex);
			auto wait_res = this_worker.scheduled_signal.wait_for(jobs_lock, std::chrono::milliseconds(2000), [&] {return !this_worker.scheduled.empty(); });
			if (wait_res == false) {
				continue;
			}
			job = this_worker.scheduled.front();
			this_worker.scheduled.pop();
			
		}
	
		if (job == nullptr) {
			//shouldn't happen with the condition_variable?
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
	assert(on_thread >= 0);

	auto& worker = _workers.at(on_thread);
	std::unique_lock my_lock(worker.scheduled_mutex);
	worker.scheduled.push(task);
	my_lock.unlock();
	worker.scheduled_signal.notify_all();
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
