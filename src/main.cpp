// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com


#include "main.h"
#define GLEW_STATIC
#define CAM_MOVE 0.3f


//-------------------------------
//-------------MAIN--------------
//-------------------------------
/*void mouseClick(int button, int status,int x,int y);
void mouseMotion(int x, int y);
static bool r_buttonDown = false;
static bool l_buttonDown = false;
static int g_yclick = 0;
static int g_ylclick = 0;
static int g_xlclick = 0;
int mouse_old_x, mouse_old_y;
glm::vec3 originPos;
glm::vec3 originView;
*/
float fps = 0;
float preTime = 0;
float currTime;
int frames = 0;
void CalcFPS()
{
	frames ++;
	currTime = glfwGetTime();
	if(currTime - preTime > 1000)
	{
		fps = frames*1000.0/(currTime - preTime);
		preTime = currTime;
		frames = 0;
	}
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

	#ifdef __APPLE__
	// Needed in OSX to force use of OpenGL3.2 
	glfwWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	#endif 

	// Set up pathtracer stuff
	bool loadedScene = true;
	finishedRender = false;

	targetFrame = 0;
	singleFrameMode = false;

	// Load scene file
	for(int i=1; i<argc; i++){
		string header; string data;
		istringstream liness(argv[i]);
		getline(liness, header, '=');
		getline(liness, data, '=');

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
		system("PAUSE");
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

	system("PAUSE");
	return 0;
}


void mainLoop() {
	while(!glfwWindowShouldClose(window)){
		glfwPollEvents();

		runCuda();
		//CalcFPS();
		//printf("fps: %f", fps);
		string title = "GPU Pathtracer | Iterations: " + utilityCore::convertIntToString(iterations); /*+ " | FPS: " + utilityCore::convertFloatToString(fps);*/
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

		//cout<<"iterations#: "<<iterations<<endl;
		// execute the kernel for ray tracing
		//cudaRaytraceCore(dptr, renderCam, targetFrame, iterations, materials, renderScene->materials.size(), geoms, renderScene->objects.size());
		
		// execute the kernel for path tracing
		cudaPathTraceCore(dptr, renderCam, targetFrame, iterations, materials, renderScene->materials.size(), geoms, renderScene->objects.size());

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	} else {
		cout<<"2"<<endl;
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
			//gamma.divisor = renderCam->iterations;
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
	window = glfwCreateWindow(width, height, "GPU Pathtracer", NULL, NULL);
	if (!window){
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	//glfwSetMouseButtonCallback(window, mouseClick);

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

	glUseProgram(program);
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

void cameraReset(){
	iterations = 0;
	for(int i = 0; i < width * height; i++)
		renderCam->image[i] = glm::vec3(0.0f);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){	
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
	else{
		switch(key){
			case GLFW_KEY_A:
				renderCam->positions[0].x += (float)CAM_MOVE;
				cameraReset();
				break;
			case GLFW_KEY_D:
				renderCam->positions[0].x -= (float)CAM_MOVE;
				cameraReset();
				break;
			case GLFW_KEY_W:
				renderCam->positions[0].y += (float)CAM_MOVE;
				cameraReset();
				break;
			case GLFW_KEY_S:
				renderCam->positions[0].y -= (float)CAM_MOVE;
				cameraReset();
				break;
			case GLFW_KEY_Q:  //zoom in
				renderCam->positions[0].z -= (float)CAM_MOVE;
				cameraReset();
				break;
			case GLFW_KEY_E:  //zoom out
				renderCam->positions[0].z += (float)CAM_MOVE;
				cameraReset();
				break;
			default:
				break;
		}
	}	
}


//mouse click motion function 
/*void mouseClick(int button,int state, int x,int y)
{
	if(button == GLUT_RIGHT_BUTTON)
	{
		//std::cout<<"ss"<<std::endl;
		r_buttonDown = (state == GLUT_DOWN) ? true:false;
		g_yclick = y - 5.0 * renderCam->positions->z;
	}
	else if(button == GLUT_LEFT_BUTTON)
	{
		l_buttonDown = (state == GLUT_DOWN) ? true:false;
		g_ylclick = y - (renderCam->positions->y + renderCam->views->y);
		g_xlclick = x -(renderCam->positions->x + renderCam->views->x);
	}
}

void mouseMotion(int x, int y)
{
	if(r_buttonDown)
	{
		renderCam->positions->z = (y - g_yclick)/5.0;
		iterations = 0;	
		finishedRender = false;
		runCuda();
		//glutPostRedisplay();
	}
	else if(l_buttonDown)
	{
		renderCam->views->x = (x - g_xlclick)/500.0;
		renderCam->views->y = (y - g_ylclick)/500.0;
		iterations = 0;
		finishedRender = false;
		runCuda();
		//glutPostRedisplay();
	}
}*/

