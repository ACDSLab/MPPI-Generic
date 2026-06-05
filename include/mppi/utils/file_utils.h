//
// Created by jgibson37 on 1/9/20.
//

#ifndef MPPIGENERIC_FILE_UTILS_H
#define MPPIGENERIC_FILE_UTILS_H

#include <string>
#ifdef _WIN32
#include <io.h>
#define F_OK 0
#define access _access
#else
#include <unistd.h>
#endif

inline bool fileExists(const std::string& name)
{
  return (access(name.c_str(), F_OK) != -1);
}

#endif  // MPPIGENERIC_FILE_UTILS_H
