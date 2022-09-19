#include "Timer.h"
#include <iostream>

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
		std::cout << "Duration: " << _name << " = " << duration << "\n";
	}
}