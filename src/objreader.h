#include <fstream>
#include <iostream>
#include <sstream>
#include <glm/glm.hpp>
#include <vector>

using namespace std;
using namespace glm;


void OBJreader(vector<vec3> &vv,vector<vec3> &fn,vector<vec3> &vi,string file,vec3 &maxpoint,vec3 &minpoint);