#pragma once
#include <iostream>
#include <format>

class StdoutLogger {
private:
	std::string _prefix;
public:
	StdoutLogger(const std::string &prefix) : _prefix(prefix) {}
	template<class... Types> void print(const std::string &str, Types... args) {
		std::string formatted = std::vformat(str, std::make_format_args(args...));
		std::cout << "[" << _prefix << "] " << formatted << std::endl;
	}
};

inline StdoutLogger debug("DEBUG");
inline StdoutLogger trace("TRACE");

//#define ENABLE_TRACE
#define LOG_DEBUG(...) debug.print(__VA_ARGS__);

#ifdef ENABLE_TRACE
#define LOG_TRACE(...) trace.print(__VA_ARGS__);
#else
#define LOG_TRACE(...) 
#endif
