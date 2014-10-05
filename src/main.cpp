// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include "main.h"
#include "utilities.h"
#include "tiny_obj_loader.h"
#include <map>
#define GLEW_STATIC

#define NUM_OF_CAMERA_RAY_PER_PIXEL 1

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){
  // Set up pathtracer stuff
  bool loadedScene = false;
  finishedRender = false;

  targetFrame = 0;
  singleFrameMode = false;

  // Load scene file
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "scene")==0){
      renderScene = new scene(data);
      loadedScene = true;
    }else if(strcmp(header.c_str(), "frame")==0){
      targetFrame = atoi(data.c_str());
      singleFrameMode = true;
    }
  }

  if(!loadedScene){
    cout << "Error: scene file needed!" << endl;
    return 0;
  }

#if(0)//OBJ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string inputfile = argv[1];
std::vector<tinyobj::shape_t> shapes;
std::vector<tinyobj::material_t> materials;

std::string err = tinyobj::LoadObj(shapes, materials, inputfile.c_str());

if (!err.empty()) {
  std::cerr << err << std::endl;
  exit(1);
}

std::cout << "# of shapes    : " << shapes.size() << std::endl;
std::cout << "# of materials : " << materials.size() << std::endl;

for (size_t i = 0; i < shapes.size(); i++) {
  printf("shape[%ld].name = %s\n", i, shapes[i].name.c_str());
  printf("Size of shape[%ld].indices: %ld\n", i, shapes[i].mesh.indices.size());
  printf("Size of shape[%ld].material_ids: %ld\n", i, shapes[i].mesh.material_ids.size());
  assert((shapes[i].mesh.indices.size() % 3) == 0);
  for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
    printf("  idx[%ld] = %d, %d, %d. mat_id = %d\n", f, shapes[i].mesh.indices[3*f+0], shapes[i].mesh.indices[3*f+1], shapes[i].mesh.indices[3*f+2], shapes[i].mesh.material_ids[f]);
  }

  printf("shape[%ld].vertices: %ld\n", i, shapes[i].mesh.positions.size());
  assert((shapes[i].mesh.positions.size() % 3) == 0);
  for (size_t v = 0; v < shapes[i].mesh.positions.size() / 3; v++) {
    printf("  v[%ld] = (%f, %f, %f)\n", v,
      shapes[i].mesh.positions[3*v+0],
      shapes[i].mesh.positions[3*v+1],
      shapes[i].mesh.positions[3*v+2]);
  }
}

for (size_t i = 0; i < materials.size(); i++) {
  printf("material[%ld].name = %s\n", i, materials[i].name.c_str());
  printf("  material.Ka = (%f, %f ,%f)\n", materials[i].ambient[0], materials[i].ambient[1], materials[i].ambient[2]);
  printf("  material.Kd = (%f, %f ,%f)\n", materials[i].diffuse[0], materials[i].diffuse[1], materials[i].diffuse[2]);
  printf("  material.Ks = (%f, %f ,%f)\n", materials[i].specular[0], materials[i].specular[1], materials[i].specular[2]);
  printf("  material.Tr = (%f, %f ,%f)\n", materials[i].transmittance[0], materials[i].transmittance[1], materials[i].transmittance[2]);
  printf("  material.Ke = (%f, %f ,%f)\n", materials[i].emission[0], materials[i].emission[1], materials[i].emission[2]);
  printf("  material.Ns = %f\n", materials[i].shininess);
  printf("  material.Ni = %f\n", materials[i].ior);
  printf("  material.dissolve = %f\n", materials[i].dissolve);
  printf("  material.illum = %d\n", materials[i].illum);
  printf("  material.map_Ka = %s\n", materials[i].ambient_texname.c_str());
  printf("  material.map_Kd = %s\n", materials[i].diffuse_texname.c_str());
  printf("  material.map_Ks = %s\n", materials[i].specular_texname.c_str());
  printf("  material.map_Ns = %s\n", materials[i].normal_texname.c_str());
  std::map<std::string, std::string>::const_iterator it(materials[i].unknown_parameter.begin());
  std::map<std::string, std::string>::const_iterator itEnd(materials[i].unknown_parameter.end());
  for (; it != itEnd; it++) {
    printf("  material.%s = %s\n", it->first.c_str(), it->second.c_str());
  }
  printf("\n");
}
#endif
//OBJ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Set up camera stuff from loaded pathtracer settings
  iterations = 0;
  renderCam = &renderScene->renderCam;
  width = renderCam->resolution[0];
  height = renderCam->resolution[1];

  if(targetFrame >= renderCam->frames){
    cout << "Warning: Specified target frame is out of range, defaulting to frame 0." << endl;
    targetFrame = 0;
  }

  // Initialize CUDA and GL components
  if (init(argc, argv)) {
	//initialize raypool
	rayPoolSize = renderCam->resolution.x * renderCam->resolution.y;
	cudaMalloc((void **)&rayPoolA,rayPoolSize * sizeof(ray));
	cudaMalloc((void **)&rayPoolB,rayPoolSize * sizeof(ray));

	// init image to GPU
	cudaMalloc((void**)&cudaImageBuffer, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaImageBuffer, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

    // GLFW main loop
    mainLoop();
  }

  
  cudaFree(rayPoolA);
  cudaFree(rayPoolB);
  cudaFree(cudaImageBuffer);
  return 0;
}

void mainLoop() {
  while(!glfwWindowShouldClose(window)){
    glfwPollEvents();
    runCuda();

    string title = "CIS565 Render | " + utilityCore::convertIntToString(iterations) + " Iterations";
		glfwSetWindowTitle(window, title.c_str());
    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glClear(GL_COLOR_BUFFER_BIT);   

    // VAO, shader program, and texture already bound
    glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
    glfwSwapBuffers(window);
  }
  glfwDestroyWindow(window);
  glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	if(iterations < renderCam->iterations){
		uchar4 *dptr=NULL;
		iterations++;
		cudaGLMapBufferObject((void**)&dptr, pbo);
  
		// pack geom and material arrays
		geom* geoms = new geom[renderScene->objects.size()];
		material* materials = new material[renderScene->materials.size()];
    
		for (int i=0; i < renderScene->objects.size(); i++) {
			geoms[i] = renderScene->objects[i];
		}
		for (int i=0; i < renderScene->materials.size(); i++) {
			materials[i] = renderScene->materials[i];
		}
		float startTime = utilityCore::getCurrentTime();
		// execute the kernel
		cudaRaytraceCore(dptr,cudaImageBuffer, renderCam, rayPoolA, rayPoolB, rayPoolSize, targetFrame, iterations, materials, renderScene->materials.size(), geoms, renderScene->objects.size() );
		float endTime =  utilityCore::getCurrentTime();
		float len = endTime - startTime;
		if(iterations%10 ==0) cout<<1000.0f*len<<" ms"<<endl;
    
		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
		} else {

		if (!finishedRender) {
		// output image file
		image outputImage(renderCam->resolution.x, renderCam->resolution.y);

			for (int x=0; x < renderCam->resolution.x; x++) {
				for (int y=0; y < renderCam->resolution.y; y++) {
					int index = x + (y * renderCam->resolution.x);
					outputImage.writePixelRGB(renderCam->resolution.x-1-x,y,renderCam->image[index]);
			}
		}
      
		gammaSettings gamma;
		gamma.applyGamma = true;
		gamma.gamma = 1.0;
		gamma.divisor = 1.0; 
		outputImage.setGammaSettings(gamma);
		string filename = renderCam->imageName;
		string s;
		stringstream out;
		out << targetFrame;
		s = out.str();
		utilityCore::replaceString(filename, ".bmp", "."+s+".bmp");
		utilityCore::replaceString(filename, ".png", "."+s+".png");
		outputImage.saveImageRGB(filename);
		cout << "Saved frame " << s << " to " << filename << endl;
		finishedRender = true;
		if (singleFrameMode==true) {
			cudaDeviceReset(); 
			exit(0);
		}
		}
		if (targetFrame < renderCam->frames-1) {

			// clear image buffer and move onto next frame
			targetFrame++;
			iterations = 0;
			for(int i=0; i<renderCam->resolution.x*renderCam->resolution.y; i++){
			renderCam->image[i] = glm::vec3(0,0,0);
		}
		cudaDeviceReset(); 
		finishedRender = false;
		}
	}
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init(int argc, char* argv[]) {
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
      return false;
  }

  width = 800;
  height = 800;
  window = glfwCreateWindow(width, height, "CIS 565 Pathtracer", NULL, NULL);
  if (!window){
      glfwTerminate();
      return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);

  // Set up GL context
  glewExperimental = GL_TRUE;
  if(glewInit()!=GLEW_OK){
    return false;
  }

  // Initialize other stuff
  initVAO();
  initTextures();
  initCuda();
  initPBO();
  
  GLuint passthroughProgram;
  passthroughProgram = initShader();

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  return true;
}

void initPBO(){
  // set up vertex data parameter
  int num_texels = width*height;
  int num_values = num_texels * 4;
  int size_tex_data = sizeof(GLubyte) * num_values;
    
  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1, &pbo);

  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
  cudaGLRegisterBufferObject(pbo);

}

void initCuda(){
  cudaGLSetGLDevice(0);

  // Clean up on program exit
  atexit(cleanupCuda);
}

void initTextures(){
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
  const char *attribLocations[] = { "Position", "Texcoords" };
  GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
  GLint location;
  
  //glUseProgram(program);
  if ((location = glGetUniformLocation(program, "u_image")) != -1)
  {
    glUniform1i(location, 0);
  }

  return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    
    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char* description){
    fputs(description, stderr);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}
