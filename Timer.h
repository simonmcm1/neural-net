#pragma once

#include <chrono>

class Timer {
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> _started_at;
	bool _has_finalized = false;
	std::string _name;
public:
	Timer(const std::string &name);
	~Timer();
	void reset();
	void end();
};