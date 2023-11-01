#pragma once
/**
 * Created by Bogdan on 11/01/2023
 */

#include <cstdarg>
#include <cstdio>
#include <vector>
#include <memory>

namespace mppi
{
namespace util
{
enum class LOG_LEVEL : int
{
  DEBUG = 0,
  INFO,
  WARNING,
  ERROR,
  NONE
};

const char BLACK[] = "\033[0;30m";
const char RED[] = "\033[0;31m";
const char GREEN[] = "\033[0;32m";
const char YELLOW[] = "\033[0;33m";
const char BLUE[] = "\033[0;34m";
const char MAGENTA[] = "\033[0;35m";
const char CYAN[] = "\033[0;36m";
const char WHITE[] = "\033[0;37m";
const char RESET[] = "\033[0m";

class MPPILogger
{
public:
  MPPILogger() = default;
  MPPILogger(const MPPILogger& other) = default;
  MPPILogger(MPPILogger&& other) = default;
  virtual ~MPPILogger() = default;
  MPPILogger& operator=(const MPPILogger& other) = default;
  MPPILogger& operator=(MPPILogger&& other) = default;

  explicit MPPILogger(LOG_LEVEL level)
  {
    setLogLevel(level);
  }

  /**
   * @brief Set the Log Level
   *
   * @param level
   */
  void setLogLevel(const LOG_LEVEL& level)
  {
    log_level_ = level;
  }

  /**
   * @brief Set the Output Stream
   *
   * @param output file stream to write (stdout, stderr, nullptr, etc.)
   */
  void setOutputStream(std::FILE* const output)
  {
    output_stream_ = output;
  }

  /**
   * @brief Get the Log Level object
   *
   * @return LOG_LEVEL
   */
  LOG_LEVEL getLogLevel() const
  {
    return log_level_;
  }

  /**
   * @brief Get the Output Stream object
   *
   * @return std::FILE*
   */
  std::FILE* getOutputStream() const
  {
    return output_stream_;
  }

  /**
   * @brief       Log debug messages to the output stream in green if the log level is set for DEBUG
   * @param fmt   Format string (if additional arguments are passed) or message to display
   */
  virtual void debug(const char* fmt, ...)
  {
    if (log_level_ <= LOG_LEVEL::DEBUG)
    {
      std::va_list argptr;
      va_start(argptr, fmt);
      surround_fprintf(output_stream_, GREEN, RESET, fmt, argptr);
      va_end(argptr);
    }
  }

  /**
   * @brief       Log info messages to the output stream in cyan if the log level is set for INFO
   * @param fmt   Format string (if additional arguments are passed) or message to display
   */
  virtual void info(const char* fmt, ...)
  {
    if (log_level_ <= LOG_LEVEL::INFO)
    {
      std::va_list argptr;
      va_start(argptr, fmt);
      surround_fprintf(output_stream_, CYAN, RESET, fmt, argptr);
      va_end(argptr);
    }
  }

  /**
   * @brief       Log debug messages to the output stream in yellow if the log level is set for WARNING
   * @param fmt   Format string (if additional arguments are passed) or message to display
   */
  virtual void warning(const char* fmt, ...)
  {
    if (log_level_ <= LOG_LEVEL::WARNING)
    {
      std::va_list argptr;
      va_start(argptr, fmt);
      surround_fprintf(output_stream_, YELLOW, RESET, fmt, argptr);
      va_end(argptr);
    }
  }

  /**
   * @brief       Log debug messages to the output stream in red if the log level is set for ERROR
   * @param fmt   Format string (if additional arguments are passed) or message to display
   */
  virtual void error(const char* fmt, ...)
  {
    if (log_level_ <= LOG_LEVEL::ERROR)
    {
      std::va_list argptr;
      va_start(argptr, fmt);
      surround_fprintf(output_stream_, RED, RESET, fmt, argptr);
      va_end(argptr);
    }
  }

protected:
  LOG_LEVEL log_level_ = LOG_LEVEL::WARNING;
  std::FILE* output_stream_ = stdout;

  /**
   * @brief Prints a colored output to a provided fstream. It does this by first creating the formatted string
   * as a std::vector<char> so that it can be used as an input to fprintf with a different format string
   *
   * @param fstream   file stream to write output to
   * @param color     color code to use on provided string
   * @param fmt       format string
   * @param ...       extra variables for format string
   */
  virtual void surround_fprintf(std::FILE* fstream, const char* prefix, const char* suffix, const char* fmt,
                                std::va_list args)
  {
    // introducing a second copy of the args as calling vsnprintf leaves args in an indeterminate state
    std::va_list args_cpy;
    va_copy(args_cpy, args);
    // figure out size of formatted string, also uses up args
    std::vector<char> buf(1 + std::vsnprintf(nullptr, 0, fmt, args));
    // Fill buffer with formatted string using copy of the args
    std::vsnprintf(buf.data(), buf.size(), fmt, args_cpy);
    va_end(args_cpy);
    // print formatted string but colored
    std::fprintf(fstream, "%s%s%s", prefix, buf.data(), suffix);
  }
};

using MPPILoggerPtr = std::shared_ptr<MPPILogger>;
}  // namespace util
}  // namespace mppi
