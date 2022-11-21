#pragma once
#include "util.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

class Task {
private:
	int _thread_index = -1;
	bool _is_complete = false;
public:
	int requested_thread_index = -1;
	batch_function& task;
	size_t data_index;
	size_t data_len;

	std::condition_variable is_complete_cv;
	std::mutex is_complete_mutex;

	Task() = delete;
	Task(batch_function& func, size_t index, size_t len) :
		task(func), data_index(index), data_len(len) {}

	bool is_complete() { return _is_complete; }
	void mark_complete();
	void wait_for_complete();

	int get_thread_index() { return _thread_index; }
	void set_thread_index(int thread_index) { _thread_index = thread_index; }
};

struct Worker {
	std::thread thread;
	std::mutex scheduled_mutex;
	std::condition_variable scheduled_signal;
	std::queue<Task*> scheduled;
	bool terminate = false;
};

class ThreadPool {
private:
	int _nthreads;
	std::vector<Worker> _workers;
	void SchedulerLoop(int thread_index);

	//static ThreadPool global_instance;
public:
	//static ThreadPool& instance() { return global_instance; }
	ThreadPool() = delete;
	ThreadPool(int nthreads);
	~ThreadPool();

	int nthreads() { return _nthreads; }

	void schedule(Task *task, int on_thread);
	void batch_jobs(batch_function& task, size_t data_len);

};