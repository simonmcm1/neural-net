#pragma once
#include <iostream>
#include <fmt/core.h>

class StdoutLogger {
private:
	std::string _prefix;
public:
	StdoutLogger(const std::string &prefix) : _prefix(prefix) {}
	template<class... Types> void print(const std::string &str, Types... args) {
		std::string formatted = fmt::vformat(str, fmt::make_format_args(args...));
		std::cout << "[" << _prefix << "] " << formatted << std::endl;
	}
};

class Logs {
public:
	static StdoutLogger debug;
	static StdoutLogger trace;
};

//#define ENABLE_TRACE

#define LOG_DEBUG(...) Logs::debug.print(__VA_ARGS__);

#ifdef ENABLE_TRACE
#define LOG_TRACE(...) Logs::trace.print(__VA_ARGS__);
#else
#define LOG_TRACE(...) 
#endif
