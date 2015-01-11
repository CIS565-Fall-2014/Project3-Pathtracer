// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include "main.h"
#include <cstring>
#define GLEW_STATIC
#pragma   comment(lib,"FreeImage.lib")

__host__ __device__ glm::vec3 multiplyMVMain(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){
  // Set up pathtracer stuff
  bool loadedScene = true;
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

  // Set up camera stuff from loaded pathtracer settings
  iterations = 0;
  renderCam = &renderScene->renderCam;
  width = renderCam->resolution[0];
  height = renderCam->resolution[1];

  if(targetFrame >= renderCam->frames){
    cout << "Warning: Specified target frame is out of range , defaulting to frame 0." << endl;
    targetFrame = 0;
  }

  // Initialize CUDA and GL components
  if (init(argc, argv)) {
    // GLFW main loop
    mainLoop();
  }

  return 0;
}
cudaEvent_t start, stop;
float timeDuration;
void mainLoop() {
  while(!glfwWindowShouldClose(window)){
    glfwPollEvents();

	//cuda timer event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

    runCuda();

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &timeDuration, start, stop );

    string title = "GPU MC Pathtracer | " + utilityCore::convertIntToString(iterations) + " Iterations | " + 
		utilityCore::convertFloatToString(timeDuration) + "ms";
	glfwSetWindowTitle(window, title.c_str());
	//char title[1000];
	//sprintf(title, "GPU Path Tracer | %d iterations", iterations);
	//glfwSetWindowTitle(window, title);
    
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
  
    // execute the kernel
    cudaRaytraceCore(dptr, renderCam, targetFrame, iterations, 
		materials, renderScene->materials.size(), geoms, renderScene->objects.size(), 
		textureColor, textureMap );
    
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
      //gamma.divisor = 1.0; 
	  gamma.divisor = renderCam->iterations;
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
  window = glfwCreateWindow(width, height, "GPU MC Pathtracer", NULL, NULL);
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
//initMesh();
  initCuda();
  initPBO();
  
  //initialize texture
  initTextureMap(-1, "C:/Users/AppleDu/Documents/GitHub/Project3-Pathtracer/data/texture/wood2.jpg");
  initTextureMap(-2, "C:/Users/AppleDu/Documents/GitHub/Project3-Pathtracer/data/texture/earthmap1024.png");
  initTextureMap(-3, "C:/Users/AppleDu/Documents/GitHub/Project3-Pathtracer/data/texture/checker.jpg");
  initTextureMap(-4, "C:/Users/AppleDu/Documents/GitHub/Project3-Pathtracer/data/texture/wall.jpg");
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
	// Use device with highest Gflops/s
 // cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );
	//initMesh();
	//cudaDeviceSynchronize();  //Blocks until the device has completed all preceding requested tasks
	//cudaDeviceReset(); //Explicitly destroys and cleans up all resources associated with the current device in the current process
	//initMesh();
	cudaGLSetGLDevice(0);  //Records the calling thread's current OpenGL context as the OpenGL context to use for OpenGL interoperability with the CUDA device

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


void initMesh(){   //intialize cuda memory for the triangle list in MESH objects

	for(int i = 0; i < renderScene->objects.size(); i++){
		if(renderScene->objects[i].type == MESH){   
			triangle * cudatris = NULL;
			cudaMalloc((void**)&cudatris, renderScene->objects[i].numOfTris *sizeof(triangle));
			cudaMemcpy( cudatris, renderScene->objects[i].tris, renderScene->objects[i].numOfTris *sizeof(triangle), cudaMemcpyHostToDevice);
			renderScene->objects[i].tris = cudatris;
		}
	}


}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
  //deleteMesh();
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

void deleteMesh(){
	for(int i = 0; i < renderScene->objects.size(); i++){
		if(renderScene->objects[i].type == MESH){
			cudaFree(renderScene->objects[i].tris);
		}
	}
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char* description){
    fputs(description, stderr);
}


void cameraReset()
{
	iterations = 0;
	//preColors = new glm::vec3[width * height];		
	for(int i = 0; i < width * height; i++)
		renderCam->image[i] = glm::vec3(0,0,0);		
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
	renderCam = &renderScene->renderCam;
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
	else if( key ==GLFW_KEY_A ){   //left
		glm::vec3 right = glm::cross(renderCam->views[0], renderCam->ups[0]);
		renderCam->positions[0] += (float)STEP_SIZE *0.5f * right;
		//renderCam->positions[0].x -= (float)STEP_SIZE * 0.5f;
		cameraReset();
	}
	else if( key ==GLFW_KEY_D ){   //right
		glm::vec3 right = glm::cross(renderCam->views[0], renderCam->ups[0]);
		renderCam->positions[0] += -(float)STEP_SIZE * 0.5f*right;
		//renderCam->positions[0].x += (float)STEP_SIZE * 0.5f;
		cameraReset();
	}
	else if( key ==GLFW_KEY_W ){   //up
		renderCam->positions[0] += (float)STEP_SIZE *0.5f* (renderCam->ups[0]);
		//renderCam->positions[0].y += (float)STEP_SIZE * 0.5f;
		cameraReset();
	}
	else if( key ==GLFW_KEY_S ){   //down
		renderCam->positions[0] += - (float)STEP_SIZE *0.5f* (renderCam->ups[0]);
		//renderCam->positions[0].y -= (float)STEP_SIZE * 0.5f;
		cameraReset();
	}
	else if ( key ==GLFW_KEY_Q ){  //forward
		renderCam->positions[0] += (float)STEP_SIZE * 0.5f* (renderCam->views[0]);
		//renderCam->positions[0].z -= (float)STEP_SIZE * 0.5f;
		cameraReset();
	}
	else if ( key == GLFW_KEY_E ){  //backward
		renderCam->positions[0] += - (float)STEP_SIZE * 0.5f* (renderCam->views[0]);
		//renderCam->positions[0].z += (float)STEP_SIZE * 0.5f;
		cameraReset();
	}
	else{
		glm::vec3 translation = *renderCam->positions;
		glm::vec3 scale(1,1,1);
		glm::vec3 rotationV(0,0,0);   //view rotation
		glm::vec3 rotationU(0,0,0);   //up rotation
		if ( key == GLFW_KEY_UP){  //rotate up
			rotationV = glm::vec3((float)STEP_SIZE,0,0);   //x-axis rotation
			/*glm::vec3 translation = *renderCam->positions;
			glm::vec3 rotation((float)STEP_SIZE,0,0);   //x-axis rotation
			glm::vec3 scale(1,1,1);
			glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
			cudaMat4	transformMy = utilityCore::glmMat4ToCudaMat4(transform);
			glm::vec3 right = glm::cross(renderCam->views[0], renderCam->ups[0]);
			glm::vec3 newView = multiplyMVMain(transformMy, glm::vec4(*renderCam->views,0.0f));
			glm::vec3 newUp = glm::cross(right, newView);
			renderCam->views[0] = newView;
			renderCam->ups[0] = newUp;
			cameraReset();*/
		}
		else if ( key == GLFW_KEY_DOWN ){  //rotate down
			rotationV = glm::vec3(-(float)STEP_SIZE,0,0);   //x-axis rotation
		}
		else if(key == GLFW_KEY_LEFT){   //rotate to left
			rotationV = glm::vec3(0,-(float)STEP_SIZE,0);   //y-axis rotation
		}
		else if(key == GLFW_KEY_RIGHT){   //rotate to left
			rotationV = glm::vec3(0,(float)STEP_SIZE,0);   //y-axis rotation
		}
		else if(key == GLFW_KEY_COMMA){   //rotate CCW
			rotationU = glm::vec3(0,0,-(float)STEP_SIZE);   //z-axis rotation
		}
		else if(key == GLFW_KEY_PERIOD){   //rotate CW
			rotationU = glm::vec3(0,0,(float)STEP_SIZE);   //z-axis rotation
		}
		glm::mat4 transform;
		cudaMat4	transformMy;
		glm::vec3 right, newView, newUp;
		if(glm::length(rotationV) > 0){
			transform = utilityCore::buildTransformationMatrix(translation, rotationV, scale);
			transformMy = utilityCore::glmMat4ToCudaMat4(transform);
			right = glm::cross(renderCam->views[0], renderCam->ups[0]);
			newView = multiplyMVMain(transformMy, glm::vec4(renderCam->views[0],0.0f));
			newUp = glm::cross(right, newView);
			
		}
		else if(glm::length(rotationU) > 0){
			transform = utilityCore::buildTransformationMatrix(translation, rotationU, scale);
			transformMy = utilityCore::glmMat4ToCudaMat4(transform);
			right = glm::cross(renderCam->views[0], renderCam->ups[0]);
			newUp = multiplyMVMain(transformMy, glm::vec4(renderCam->ups[0],0.0f));
			newView = glm::cross(newUp, right);
		}
		renderCam->views[0] = newView;
		renderCam->ups[0] = newUp;
		cameraReset();
	}
}

//------------------------------
//-------TEXTURE STUFF---------
//------------------------------
//http://www.mingw.org/
//http://freeimage.sourceforge.net/download.html
//https://www.opengl.org/discussion_boards/showthread.php/163929-image-loading?p=1158293#post1158293
//http://inst.eecs.berkeley.edu/~cs184/fa09/resources/sec_UsingFreeImage.pdf

//loading and initializing texture map
void initTextureMap(int id, char* textureFileName){
	int h = 0,  w = 0;
	int tmp = loadTexture(textureFileName,textureColor,h,w);
	if( tmp != -1){
		tex newTex;
		newTex.id = id;
		newTex.start = tmp;   //start index, point to textureColor
		newTex.h = h;   //height
		newTex.w = w;   //width
		textureMap.push_back(newTex);
	}
}

int loadTexture(char* file, std::vector<glm::vec3> &c, int &h,int &w){
	FIBITMAP* image = FreeImage_Load( FreeImage_GetFileType(file, 0), file);
	if(!image){
		printf("Error: fail to open texture file %s\n", file );
		FreeImage_Unload(image);
		return -1;
	}
	image = FreeImage_ConvertTo32Bits(image);
	 
	w = FreeImage_GetWidth(image);
	h = FreeImage_GetHeight(image);
	if( w == 0 && h == 0 ) {
		printf("Error: texture file is empty\n");
		FreeImage_Unload(image);
		return -1;
	}

	int start = c.size();
	//int total = w * h;
	//if(n.size()>0)  //useful when load multiple picture of texture
		//total += n[n.size()-1];
	//n.push_back(total);

	//int k = 0;
	for(int i = 0; i < w; i++){
	   for(int j = 0;j < h; j++){
		   RGBQUAD color;
		   FreeImage_GetPixelColor( image, i, j, &color );
		   glm::vec3 nc(color.rgbRed, color.rgbGreen, color.rgbBlue);
	       c.push_back(nc);
	
		   //printf("color @ %d is %.2f, %.2f, %.2f\n",k, c[k].r, c[k].g, c[k].b);
		  // k++;
	   }
	}
	
	FreeImage_Unload(image);
	printf("Loaded texture %s with %dx%d pixels\n", file,w,h );
	return start;
}



