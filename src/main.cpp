// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include "main.h"
#define GLEW_STATIC

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
	scenename = data.substr(0,data.length()-4);
	scenename = scenename.substr(7,scenename.length());
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

  // Set up camera stuff from loaded pathtracer settings
  iterations = 0;
  renderCam = &renderScene->renderCam;
  width = (int)renderCam->resolution[0];
  height = (int)renderCam->resolution[1];

  if(targetFrame >= renderCam->frames){
    cout << "Warning: Specified target frame is out of range, defaulting to frame 0." << endl;
    targetFrame = 0;
  }

  // Initialize CUDA and GL components
  if (init(argc, argv)) {
    // GLFW main loop
    mainLoop();
  }

  return 0;
}

void mainLoop() {
  while(!glfwWindowShouldClose(window)){
    glfwPollEvents();
    runCuda();
	theFpsTracker.timestamp();
	string FPS;
	stringstream ss;
	ss<<theFpsTracker.fpsAverage();
	ss>>FPS;
    string title = "CIS565 Render | " + utilityCore::convertIntToString(iterations) + " Iterations" + "  AverageFPS:" + FPS;
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
  if(iterations < (int)renderCam->iterations){
	if(isRecording||iterations == (int)renderCam->iterations-1)
		 grabScreen();
    uchar4 *dptr=NULL;
    iterations++;
    cudaGLMapBufferObject((void**)&dptr, pbo);
  
    // pack geom and material arrays
    geom* geoms = new geom[renderScene->objects.size()];
    material* materials = new material[renderScene->materials.size()];
    

    for (int i=0; i < (int)renderScene->objects.size(); i++) {
      geoms[i] = renderScene->objects[i];
    }
    for (int i=0; i < (int)renderScene->materials.size(); i++) {
      materials[i] = renderScene->materials[i];
    }

    // execute the kernel
    cudaRaytraceCore(dptr, renderCam, targetFrame, iterations, materials, renderScene->materials.size(), 
		geoms, renderScene->objects.size() ,renderScene->colors,renderScene->lastnum,renderScene->bump_colors,renderScene->bump_lastnum);
    
    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);
  } else {

    if (!finishedRender) {
      // output image file
      image outputImage((int)renderCam->resolution.x, (int)renderCam->resolution.y);

      for (int x=0; x < renderCam->resolution.x; x++) {
        for (int y=0; y < renderCam->resolution.y; y++) {
          int index = x + (y * (int)renderCam->resolution.x);
          outputImage.writePixelRGB((int)renderCam->resolution.x-1-x,y,renderCam->image[index]);
        }
      }
      
      gammaSettings gamma;
      gamma.applyGamma = true;
      gamma.gamma = (int)1.0;
      gamma.divisor = (int)1.0; 
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

void ClearScreen()
{
	iterations = 0;
	for(int i=0; i<renderCam->resolution.x*renderCam->resolution.y; i++){
		renderCam->image[i] = glm::vec3(0,0,0);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

	if (key == GLFW_KEY_R&& action == GLFW_PRESS) 
		isRecording = !isRecording;


	if(key==GLFW_KEY_T && action == GLFW_PRESS)
	{
		ClearScreen();
		texturemap_b =!texturemap_b;
		if(texturemap_b)
			cout<<"Texture Map Enabled!"<<endl;
		else
			cout<<"Texture Map Disabled!"<<endl;
	}

	if(key==GLFW_KEY_N  && action == GLFW_PRESS)
	{
		ClearScreen();
		bumpmap_b =!bumpmap_b;
		if(bumpmap_b)
			cout<<"Bump Map(Normal Map) Enabled!"<<endl;
		else
			cout<<"Bump Map(Normal Map) Disabled!"<<endl;
	}

	if(key==GLFW_KEY_M  && action == GLFW_PRESS)
	{
		ClearScreen();
		MB_b =!MB_b;
		if(MB_b)
			cout<<"Motion Blur Enabled!"<<endl;
		else
			cout<<"Motion Blur Disabled!"<<endl;
	}

	if(key==GLFW_KEY_SPACE  && action == GLFW_PRESS)
	{
		ClearScreen();
		streamcompact_b =!streamcompact_b;
		if(streamcompact_b)
			cout<<"Stream Compaction Enabled!"<<endl;
		else
			cout<<"Stream Compaction Disabled!"<<endl;
	}

	if(key==GLFW_KEY_D && action == GLFW_PRESS)
	{
		ClearScreen();
		DOF_b =!DOF_b;
		if(DOF_b)
			cout<<"Depth of field Enabled!"<<endl;
		else
			cout<<"Depth of field Disabled!"<<endl;
	}

	if(key==GLFW_KEY_UP)
	{
		ClearScreen();
		renderCam->positions[0]=glm::vec3(renderCam->positions[0].x,renderCam->positions[0].y+0.1f,renderCam->positions[0].z);
	}
	else if(key==GLFW_KEY_DOWN)
	{
		ClearScreen();
		renderCam->positions[0]= glm::vec3(renderCam->positions[0].x,renderCam->positions[0].y-0.1f,renderCam->positions[0].z);
	}
	else if(key==GLFW_KEY_LEFT)
	{
		ClearScreen();
		renderCam->positions[0]= glm::vec3(renderCam->positions[0].x+0.1f,renderCam->positions[0].y,renderCam->positions[0].z);
	}
	else if(key==GLFW_KEY_RIGHT)
	{
		ClearScreen();
		renderCam->positions[0]= glm::vec3(renderCam->positions[0].x-0.1f,renderCam->positions[0].y,renderCam->positions->z);
	}
	else if(key==GLFW_KEY_Z)
	{
		ClearScreen();
		renderCam->positions[0]= glm::vec3(renderCam->positions[0].x,renderCam->positions[0].y,renderCam->positions[0].z-0.1f);
	}
	else if(key==GLFW_KEY_C)
	{
		ClearScreen();
		renderCam->positions[0]= glm::vec3(renderCam->positions[0].x,renderCam->positions[0].y,renderCam->positions[0].z+0.1f);
	}
}


//Added
void grabScreen(void)
{
	int window_width = 800;
	int window_height = 800;
	unsigned char* bitmapData = new unsigned char[3 * window_width * window_height];

	for (int i=0; i < window_height; i++) 
	{
		glReadPixels(0, i, window_width, 1, GL_RGB, GL_UNSIGNED_BYTE, 
			bitmapData + (window_width * 3 * ((window_height - 1) - i)));
	}

	char anim_filename[2048];
	string f1 = "output/" + scenename;
	f1 = f1 + "_%04d.png";
	char* filename = (char*)f1.c_str();
	sprintf_s(anim_filename, 2048, filename, iterations);
	stbi_write_png(anim_filename, window_width, window_height, 3, bitmapData, window_width * 3);

	delete [] bitmapData;
}