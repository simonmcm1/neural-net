#pragma once

#include <chrono>
#include <unordered_map>

class Timer {
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> _started_at;
	bool _has_finalized = false;
	std::string _name;
	static std::unordered_map<std::string, double> total_times;
public:
	Timer(const std::string &name);
	~Timer();
	void reset();
	void end();
	
	static void print_usage_report();
};
