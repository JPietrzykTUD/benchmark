#ifndef TOGATHERORNOT_INCLUDE_EXEC_HPP
#define TOGATHERORNOT_INCLUDE_EXEC_HPP
#include <istream>
#include <streambuf>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

class executor {
   protected:
      std::string output;
      std::array< char, 128 > buffer;
   public:
      executor(): output{}, buffer{} { }
   public:
      void exec( char const * command ) {
         this->output = "";
         std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command, "r"), pclose);
         if (!pipe) {
               throw std::runtime_error("popen() failed!");
            }
         while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
               this->output += buffer.data();
            }
      }
      std::string & get_output() {
         return output;
      }

};

#endif //TOGATHERORNOT_INCLUDE_EXEC_HPP
