#pragma once
#include "util.h"
#include <thread>
#include <queue>
#include <mutex>

class Task {
private:
	bool _is_complete = false;
	int _thread_index = -1;

public:
	int requested_thread_index = -1;
	batch_function& task;
	size_t data_index;
	size_t data_len;

	Task() = delete;
	Task(batch_function& func, size_t index, size_t len) :
		task(func), data_index(index), data_len(len) {}

	bool is_complete() { return _is_complete; }
	void mark_complete() { _is_complete = true; }

	int get_thread_index() { return _thread_index; }
	void set_thread_index(int thread_index) { _thread_index = thread_index; }
};

class ThreadPool {
private:
	int _nthreads;
	std::queue<Task *> _scheduled;
	std::mutex _scheduled_jobs_mutex;

	std::vector<std::thread> _threads;
	void SchedulerLoop(int thread_index);

public:
	ThreadPool() = delete;
	ThreadPool(int nthreads);
	~ThreadPool();

	int nthreads() { return _nthreads; }

	void schedule(Task *task, int on_thread);
	void batch_jobs(batch_function& task, size_t data_len);

};