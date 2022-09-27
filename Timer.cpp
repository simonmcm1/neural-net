#include "Timer.h"
#include <iostream>
#include "Logging.h"

std::unordered_map<std::string, double> Timer::total_times;

Timer::Timer(const std::string& name) :
	_name(name)
{
	reset();
}

Timer::~Timer()
{
	end();
}

void Timer::reset()
{
	_has_finalized = false;
	_started_at = std::chrono::high_resolution_clock::now();
}

void Timer::end()
{
	if (!_has_finalized) {
		_has_finalized = true;
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = duration_cast<std::chrono::milliseconds>(end - _started_at);
		//std::cout << "Duration: " << _name << " = " << duration << "\n";

		if (Timer::total_times.find(_name) == Timer::total_times.end()) {
			Timer::total_times.insert({ _name, 0.0 });
		}
		Timer::total_times[_name] += (double)duration.count();		
	}
}

void Timer::print_usage_report()
{
	LOG_DEBUG("Usages:")
	for (auto& item : total_times) {
		LOG_DEBUG("{}: {}ms", item.first, item.second);
	}
}